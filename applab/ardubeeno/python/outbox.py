"""Local bee-passage history and its upload to the ArduBeeno receiver (docs/UPLOADS.md).

Every completed bee track becomes one `bee_passage` event, healthy recording time becomes `coverage` events, and
the app's status is sent as `heartbeat` events. Each event is committed to a local SQLite database together with
its outbox row before anything is sent; a background thread uploads pending events in batches and marks exactly
the acknowledged ones. Acknowledged events stay in the local history.

Turned on by ARDUBEENO_SERVER_URL, ARDUBEENO_DEVICE_ID and ARDUBEENO_DEVICE_TOKEN (from the environment, or
KEY=value lines in data/ardubeeno.env in the app folder). Optional: ARDUBEENO_SOURCE (field, screen_demo or
replay), ARDUBEENO_OUTBOX_DB (database path), ARDUBEENO_CLOCK_TRUSTED (1/0, overrides the NTP check).
Standard library only. Nothing here runs under the monitor's lock or in the camera loop: the loop only puts
events on a queue.
"""
import ctypes
import json
import os
import queue
import random
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DATA = Path(__file__).parent.parent / "data"
METHOD = "bee_tracks_v1"
SOURCES = ("field", "screen_demo", "replay")
BATCH, MAX_BODY = 100, 2 * 1024 * 1024
COVERAGE_EVERY, HEARTBEAT_EVERY, FRAME_GAP = 15.0, 20.0, 5.0  # seconds; a longer pause in frames ends coverage
LEASE = 120.0
STOP = object()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, boot_id TEXT, started_mono REAL NOT NULL, started_wall REAL NOT NULL,
    clock_trusted INTEGER NOT NULL, source TEXT NOT NULL, revision TEXT, next_sequence INTEGER NOT NULL,
    ended_wall REAL);
CREATE TABLE IF NOT EXISTS clock_anchors (
    id INTEGER PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions, boot_id TEXT,
    mono REAL NOT NULL, wall REAL NOT NULL, evidence TEXT);
CREATE TABLE IF NOT EXISTS local_events (
    event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions, sequence INTEGER NOT NULL,
    kind TEXT NOT NULL, mono REAL NOT NULL, start_mono REAL, wall REAL NOT NULL, source TEXT NOT NULL,
    value TEXT NOT NULL, track_id INTEGER, info TEXT, occurred_at TEXT, wire TEXT,
    UNIQUE (session_id, sequence));
CREATE TABLE IF NOT EXISTS track_finalizations (
    session_id TEXT NOT NULL, track_id INTEGER NOT NULL, event_id TEXT NOT NULL UNIQUE REFERENCES local_events,
    PRIMARY KEY (session_id, track_id));
CREATE TABLE IF NOT EXISTS outbox (
    event_id TEXT PRIMARY KEY REFERENCES local_events,
    state TEXT NOT NULL CHECK (state IN ('blocked_time', 'pending', 'leased', 'acked', 'quarantined')),
    attempts INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_until REAL, last_error TEXT, acked_at REAL);
CREATE INDEX IF NOT EXISTS outbox_state ON outbox (state);
CREATE TABLE IF NOT EXISTS local_meta (key TEXT PRIMARY KEY, value TEXT);
"""


def stamp(wall):
    """RFC3339 UTC with microseconds, as the receiver stores it."""
    return datetime.fromtimestamp(wall, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def clock_trusted():
    """Is the wall clock synchronised (NTP)? Same check as timedatectl: adjtimex() != TIME_ERROR."""
    forced = os.environ.get("ARDUBEENO_CLOCK_TRUSTED")
    if forced in ("0", "1"):
        return forced == "1"
    if not sys.platform.startswith("linux"):
        return True
    try:
        buf = ctypes.create_string_buffer(512)  # struct timex, modes = 0: read only
        return ctypes.CDLL(None, use_errno=True).adjtimex(buf) != 5
    except (OSError, AttributeError):
        return True


def load_config():
    """Settings from the environment, falling back to data/ardubeeno.env (KEY=value lines)."""
    conf = {}
    try:
        for line in (DATA / "ardubeeno.env").read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and not key.startswith("#"):
                conf[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    return {k[len("ARDUBEENO_"):].lower(): os.environ.get(k) or conf.get(k) for k in
            ("ARDUBEENO_SERVER_URL", "ARDUBEENO_DEVICE_ID", "ARDUBEENO_DEVICE_TOKEN", "ARDUBEENO_SOURCE",
             "ARDUBEENO_OUTBOX_DB")}


class Store:
    """The device's SQLite history: one connection per thread, short transactions, never across network I/O."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tx = self.connect()
        tx.db.execute("PRAGMA journal_mode=WAL")
        tx.db.executescript("BEGIN IMMEDIATE;" + SCHEMA + "COMMIT;")
        tx.close()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None, check_same_thread=False)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        return Tx(db)


class Tx:
    """`with tx as db:` runs one BEGIN IMMEDIATE ... COMMIT (ROLLBACK on error)."""

    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc, *_):
        try:
            self.db.execute("ROLLBACK" if exc else "COMMIT")
        except sqlite3.Error:
            if self.db.in_transaction:  # a failed COMMIT (e.g. disk full) must not leave the transaction open
                self.db.execute("ROLLBACK")
            raise

    def close(self):
        self.db.close()


class Recorder:
    """Records events durably on a writer thread and uploads them on an uploader thread."""

    @classmethod
    def from_env(cls, source="field", revision=None, source_override=False):
        """A Recorder when the receiver is configured, else None (the app then runs without uploads)."""
        c = load_config()
        if not (c["server_url"] or c["device_id"] or c["device_token"]):
            return None
        if not (c["server_url"] and c["device_id"] and c["device_token"]):
            print("upload off: set all of ARDUBEENO_SERVER_URL, ARDUBEENO_DEVICE_ID and ARDUBEENO_DEVICE_TOKEN")
            return None
        source = source if (source_override and source) else (c["source"] or source)
        if source not in SOURCES:
            print(f"upload off: ARDUBEENO_SOURCE must be one of {', '.join(SOURCES)}")
            return None
        try:
            return cls(c["server_url"], c["device_id"], c["device_token"], source,
                       c["outbox_db"] or DATA / "outbox.sqlite3", revision)
        except (sqlite3.Error, OSError) as e:
            print(f"upload off: local history database not usable: {e}")
            return None

    def __init__(self, server_url, device_id, token, source, db_path, revision=None):
        self.server_url = server_url.rstrip("/")
        self.url = self.server_url + "/api/v1/ingest/batch"
        self.device_id, self.token, self.source = device_id, token, source
        self.store = Store(db_path)
        self.session_id, self.boot = str(uuid.uuid4()), boot_id()
        self.trusted = clock_trusted()
        self.queue, self.wake, self.lock = queue.Queue(), threading.Event(), threading.Lock()
        self.drain_cv, self.close_wake, self.stop_upload = threading.Condition(), threading.Event(), threading.Event()
        self.drain_gen, self.drained_gen = 0, -1
        self.closing, self.close_failed = False, False
        self.cov_start, self.last_frame, self.recording = None, None, False
        self.status = {"enabled": True, "server_url": self.server_url, "device_id": device_id, "source": source,
                       "session_id": self.session_id, "fault": None, "unrecorded": 0, "last_ack": None,
                       "failures": 0, "upload_error": None}
        self.db = self.store.connect()
        with self.db as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,1,NULL)",
                       (self.session_id, self.boot, time.monotonic(), time.time(), int(self.trusted), source,
                        json.dumps(revision) if revision else None))
            if self.trusted:
                self.anchor(db)
        self.initial_counts = self._read_baseline_counts(self.db.db)
        self.refresh()
        self.writer = threading.Thread(target=self.write_loop, name="outbox-writer", daemon=True)
        self.uploader = threading.Thread(target=self.upload_loop, name="outbox-uploader", daemon=True)
        self.writer.start()
        self.uploader.start()
        print(f"uploading to {self.server_url} as {device_id} ({source}), history in {db_path}")

    def _baseline_key(self):
        return f"baseline_rowid:{self.source}"

    def _read_baseline_counts(self, db):
        row = db.execute("SELECT value FROM local_meta WHERE key=?", (self._baseline_key(),)).fetchone()
        base_rowid = int(row[0]) if row and str(row[0]).isdigit() else 0
        rows = db.execute("SELECT value, info FROM local_events WHERE kind='bee_passage' AND source=? AND rowid>?",
                          (self.source, base_rowid)).fetchall()
        seen = mites = clean = checked = 0
        for v, raw in rows:
            try:
                meta = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                meta = {}
            if meta.get("pre_reset"):
                continue
            seen += 1
            mites += v == "mites_detected"
            clean += v == "not_detected"
            try:
                checked += int(meta.get("checks", 0)) > 0
            except (ValueError, TypeError):
                pass
        return {"bees_seen": seen, "bees_checked": checked, "bees_clean": clean,
                "bees_uncertain": seen - mites - clean, "count": mites}

    # --- called from the camera loop: cheap, never touch the database ---

    def frame(self, recording):
        """Every processed frame: recording=True while bee tracks are being assessed (multi mode)."""
        now = time.monotonic()
        with self.lock:
            gap = self.last_frame is not None and now - self.last_frame > FRAME_GAP
            if self.cov_start is not None and (gap or not recording):
                self.queue.put(("coverage", self.last_frame, {"start": self.cov_start}))
                self.cov_start = None
            if recording and self.cov_start is None and self.status["fault"] != "storage":
                self.cov_start = now
            self.last_frame, self.recording = now, recording

    def passage(self, track_id, assessment, info):
        now = time.monotonic()
        with self.lock:
            if self.cov_start is not None and self.status["fault"] != "storage" and now >= self.cov_start:
                end = now + 0.001  # half-open [start, end) on the server includes this passage immediately
                self.queue.put(("coverage", end, {"start": self.cov_start}))
                self.cov_start = end
        self.queue.put(("bee_passage", now, {"track_id": track_id, "value": assessment, "info": info}))

    def reset_baseline(self):
        """Advance the local display baseline without deleting events or resetting sequences."""
        self.queue.put(("reset_baseline", time.monotonic(), {}))

    def update_revision(self, revision):
        self.queue.put(("revision", time.monotonic(), {"revision": revision}))

    def _wait_drain(self, deadline):
        rem = deadline - time.monotonic()
        if rem <= 0.05:
            return
        with self.drain_cv:
            self.drain_gen += 1
            target = self.drain_gen
        self.close_wake.set()
        self.wake.set()
        with self.drain_cv:
            self.drain_cv.wait_for(
                lambda: self.drained_gen >= target or self.close_failed or self.stop_upload.is_set(),
                timeout=rem,
            )

    def close(self, timeout=10, drain=3.0):
        """Close coverage, drain pending passages/coverage if reachable, then commit and send an idle heartbeat."""
        deadline = time.monotonic() + drain
        now = time.monotonic()
        self.closing = True
        with self.lock:
            self.recording = False
            if self.cov_start is not None:
                end = min(now, (self.last_frame or now) + FRAME_GAP)
                if end > self.cov_start:
                    self.queue.put(("coverage", end, {"start": self.cov_start}))
                self.cov_start = None
        committed = threading.Event()
        self.queue.put(("barrier", now, {"event": committed}))
        committed.wait(timeout)

        self._wait_drain(deadline)

        self.queue.put(("heartbeat", time.monotonic(), {"value": "idle"}))
        self.queue.put(STOP)
        self.writer.join(timeout)

        if not self.close_failed:
            self._wait_drain(deadline)
        self.stop_upload.set()
        self.close_wake.set()
        self.wake.set()
        self.uploader.join(0.5)

    # --- writer thread ---

    def write_loop(self):
        now = time.monotonic()
        next_cov, next_beat = now + COVERAGE_EVERY, now + 2  # a first heartbeat shortly after startup
        while True:
            try:
                item = self.queue.get(timeout=1)
            except queue.Empty:
                item = None
            if item is STOP:
                self.write(lambda db: db.execute("UPDATE sessions SET ended_wall=? WHERE session_id=?",
                                                 (time.time(), self.session_id)))
                return
            if item:
                self.record(*item)
            now = time.monotonic()
            if now >= next_cov:
                next_cov = now + COVERAGE_EVERY
                with self.lock:  # close the interval so far, the next one starts where it ended
                    if self.cov_start is not None and now - self.last_frame <= FRAME_GAP and now > self.cov_start:
                        self.queue.put(("coverage", now, {"start": self.cov_start}))
                        self.cov_start = now
                if not self.trusted and clock_trusted():
                    self.write(self.seal_blocked)
                self.refresh()
            if now >= next_beat and not self.closing:
                next_beat = now + HEARTBEAT_EVERY
                self.heartbeat()

    def heartbeat(self):
        with self.lock:
            fresh = self.last_frame is not None and time.monotonic() - self.last_frame <= FRAME_GAP
            status = "storage_error" if self.status["fault"] == "storage" else \
                ("recording" if self.recording else "idle") if fresh else \
                "camera_error" if self.last_frame is not None else "idle"
        self.queue.put(("heartbeat", time.monotonic(), {"value": status}))

    def write(self, fn):
        try:
            with self.db as db:
                fn(db)
            if self.status["fault"] == "storage":
                self.status["fault"] = None
            return True
        except (sqlite3.Error, OSError) as e:
            print(f"local history write failed: {e}")
            self.status["fault"] = "storage"
            self.status["unrecorded"] += 1
            with self.lock:  # no durable recording, no coverage
                self.cov_start = None
            return False

    def record(self, kind, mono, d):
        """Allocate the sequence, insert the event (+ track mapping) and its outbox row in one transaction."""
        if kind == "barrier":
            d["event"].set()
            return
        if kind == "reset_baseline":
            self.write(lambda db: db.execute(
                "INSERT INTO local_meta VALUES (?, COALESCE((SELECT CAST(MAX(rowid) AS TEXT) FROM local_events), '0')) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (self._baseline_key(),)))
            return
        if kind == "revision":
            self.write(lambda db: db.execute("UPDATE sessions SET revision=? WHERE session_id=?",
                                             (json.dumps(d["revision"]), self.session_id)))
            return
        if kind == "coverage" and mono - d["start"] < 0.001:
            return

        def tx(db):
            if d.get("track_id") is not None and db.execute(
                    "SELECT 1 FROM track_finalizations WHERE session_id=? AND track_id=?",
                    (self.session_id, d["track_id"])).fetchone():
                return  # already finalized: keep the existing event
            seq = db.execute("SELECT next_sequence FROM sessions WHERE session_id=?", (self.session_id,)).fetchone()[0]
            db.execute("UPDATE sessions SET next_sequence=? WHERE session_id=?", (seq + 1, self.session_id))
            value = d["value"] if kind != "coverage" else "healthy"
            info = dict(d.get("info") or {})
            if kind == "heartbeat":
                info["pending_events"] = db.execute("SELECT COUNT(*) FROM outbox WHERE state IN "
                                                    "('pending','leased','blocked_time')").fetchone()[0]
            event_id = str(uuid.uuid4())
            db.execute("INSERT INTO local_events VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,NULL)",
                       (event_id, self.session_id, seq, kind, mono, d.get("start"), time.time(), self.source, value,
                        d.get("track_id"), json.dumps(info) if info else None))
            if d.get("track_id") is not None:
                db.execute("INSERT INTO track_finalizations VALUES (?,?,?)", (self.session_id, d["track_id"], event_id))
            db.execute("INSERT INTO outbox (event_id, state) VALUES (?, 'blocked_time')", (event_id,))
            if self.trusted:
                self.seal(db, event_id, time.time() - time.monotonic())

        if self.write(tx):
            self.refresh()
            self.wake.set()

    def seal(self, db, event_id, offset):
        """Map monotonic times to UTC with `offset`, store the wire JSON once and release the event for upload."""
        r = db.execute("SELECT session_id, sequence, kind, mono, start_mono, source, value, info FROM local_events "
                       "WHERE event_id=?", (event_id,)).fetchone()
        session_id, seq, kind, mono, start, source, value, info = r
        occurred = stamp(offset + mono)
        e = {"event_id": event_id, "session_id": session_id, "sequence": seq, "kind": kind, "occurred_at": occurred,
             "source": source, "method": METHOD}
        if kind == "bee_passage":
            e["assessment"] = value
        elif kind == "coverage":
            e["start_at"], e["end_at"] = stamp(offset + start), occurred
        else:
            e["status"], e["pending_events"] = value, json.loads(info)["pending_events"]
        db.execute("UPDATE local_events SET occurred_at=?, wire=? WHERE event_id=? AND wire IS NULL",
                   (occurred, json.dumps(e, separators=(",", ":")), event_id))
        db.execute("UPDATE outbox SET state='pending' WHERE event_id=? AND state='blocked_time'", (event_id,))

    def anchor(self, db):
        mono, wall = time.monotonic(), time.time()
        db.execute("INSERT INTO clock_anchors (session_id, boot_id, mono, wall, evidence) VALUES (?,?,?,?,?)",
                   (self.session_id, self.boot, mono, wall, "adjtimex synchronised"))
        return wall - mono

    def seal_blocked(self, db):
        """The clock became trusted: seal this boot's events that were waiting for it. Other boots stay blocked."""
        self.trusted = True
        offset = self.anchor(db)
        for (event_id,) in db.execute(
                "SELECT e.event_id FROM local_events e JOIN outbox o USING (event_id) JOIN sessions s USING "
                "(session_id) WHERE o.state='blocked_time' AND (s.session_id=? OR s.boot_id=?)",
                (self.session_id, self.boot)).fetchall():
            self.seal(db, event_id, offset)

    def refresh(self):
        """Counts for the page and the log: pending, blocked, quarantined, free space."""
        try:
            counts = dict(self.db.db.execute("SELECT state, COUNT(*) FROM outbox GROUP BY state").fetchall())
            oldest = self.db.db.execute("SELECT MIN(e.occurred_at) FROM outbox o JOIN local_events e USING "
                                        "(event_id) WHERE o.state IN ('pending','leased')").fetchone()[0]
            free = shutil.disk_usage(self.store.path.parent).free
        except (sqlite3.Error, OSError):
            return
        self.status.update({s: counts.get(s, 0) for s in ("pending", "leased", "blocked_time", "acked", "quarantined")})
        self.status.update(oldest_pending=oldest, db_bytes=self.store.path.stat().st_size, free_bytes=free,
                           clock_trusted=self.trusted)
        if free < 50 * 1024 * 1024:
            print(f"warning: only {free // 1024 // 1024} MB left for the local history")

    # --- uploader thread ---

    def upload_loop(self):
        db = self.store.connect()
        delay, size, failing = 0.0, BATCH, False
        while not self.stop_upload.is_set():
            if delay and not self.closing:  # backing off: new events do not shorten the wait, except final drain
                self.close_wake.wait(delay)
            else:
                self.wake.wait(0.2 if self.closing else 30)
            if self.stop_upload.is_set():
                return
            self.wake.clear()
            while not self.stop_upload.is_set():
                with self.drain_cv:
                    gen = self.drain_gen
                result, sent = self.upload_once(db, size)
                if result == "empty":
                    delay, size = 0.0, BATCH
                    with self.drain_cv:
                        self.drained_gen = max(self.drained_gen, gen)
                        self.drain_cv.notify_all()
                    break
                if result == "ok":
                    if failing and not self.closing:  # back online: a fresh heartbeat goes with the backlog
                        self.heartbeat()
                    delay, failing, self.status["failures"] = 0.0, False, 0
                    self.status["upload_error"] = None
                    if self.status["fault"] == "credentials":
                        self.status["fault"] = None
                    continue
                if result == "quarantined":  # the bad event is isolated: back to full batches
                    size = BATCH
                    continue
                failing = True
                if result == "split":  # 413 / 409 / 422 with several events: halve to isolate, same identities
                    size = max(1, sent // 2)
                    continue
                if isinstance(result, float):  # transient: exponential backoff with jitter or Retry-After
                    delay = result
                if self.closing:
                    with self.drain_cv:
                        self.close_failed = True
                        self.drain_cv.notify_all()
                    return
                break

    def upload_once(self, db, size):
        """Lease up to `size` events, send them, apply the response. Returns (result, events sent)."""
        now = time.time()
        with db as d:
            rows = d.execute(
                "SELECT o.event_id, e.wire FROM outbox o JOIN local_events e USING (event_id) "
                "WHERE o.state='pending' OR (o.state='leased' AND o.lease_until < ?) "
                "ORDER BY (e.kind='heartbeat' AND e.session_id=?) DESC, e.rowid LIMIT ?",
                (now, self.session_id, size)).fetchall()
            head = '{"schema_version":1,"device_id":%s,"events":[' % json.dumps(self.device_id)
            ids, wires, length = [], [], len(head) + 2
            for event_id, wire in rows:  # keep the encoded body within 2 MiB (the sealed JSON is ASCII)
                length += len(wire) + 1
                if length > MAX_BODY and ids:
                    break
                ids.append(event_id)
                wires.append(wire)
            if not ids:
                return "empty", 0
            d.executemany("UPDATE outbox SET state='leased', lease_owner=?, lease_until=?, attempts=attempts+1 "
                          "WHERE event_id=?", [(self.session_id, now + LEASE, i) for i in ids])
        body = (head + ",".join(wires) + "]}").encode()
        request = Request(self.url, data=body, method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        status, detail, ack, retry_after = None, "", None, ""
        try:
            with urlopen(request, timeout=2.5 if self.closing else 20) as response:
                status, ack = response.status, json.loads(response.read())
        except HTTPError as e:
            status, detail = e.code, e.read()[:500].decode("utf-8", "replace")
            retry_after = e.headers.get("Retry-After", "")
        except (OSError, ValueError) as e:  # connection errors, timeouts, invalid JSON
            detail = f"{type(e).__name__}: {e}"

        def release(state="pending", error=None, only=None):
            with db as d:
                d.executemany("UPDATE outbox SET state=?, lease_owner=NULL, lease_until=NULL, last_error=? "
                              "WHERE event_id=? AND state='leased'", [(state, error, i) for i in (only or ids)])

        if status == 200:
            ack = ack if isinstance(ack, dict) else {}
            committed = ack.get("committed_ids")
            if (ack.get("schema_version") == 1 and ack.get("device_id") == self.device_id
                    and isinstance(committed, list) and all(isinstance(i, str) for i in committed)
                    and set(committed) == set(ids)):
                with db as d:
                    d.executemany("UPDATE outbox SET state='acked', acked_at=?, lease_owner=NULL, lease_until=NULL, "
                                  "last_error=NULL WHERE event_id=?", [(time.time(), i) for i in ids])
                self.status["last_ack"] = stamp(time.time())
                self.status.update(pending=max(0, self.status.get("pending", 0) - len(ids)),
                                   acked=self.status.get("acked", 0) + len(ids))
                return "ok", len(ids)
            detail = "invalid acknowledgement"
        if status in (409, 413, 422) and len(ids) > 1:
            release(error=f"HTTP {status}: {detail}")
            return "split", len(ids)
        if status in (409, 413, 422):  # this one event is the fault: keep it, stop sending it
            release("quarantined", f"HTTP {status}: {detail}")
            print(f"upload: event {ids[0]} quarantined after HTTP {status}: {detail}")
            return "quarantined", 1
        release(error=f"HTTP {status}: {detail}" if status else detail)
        if status is not None and 400 <= status < 500:  # 401 and other client errors: configuration fault
            if self.status.get("fault") != "credentials":
                print(f"upload: HTTP {status} from the receiver, check ARDUBEENO_DEVICE_ID/TOKEN/SERVER_URL: {detail}")
            self.status["fault"] = "credentials"
            return 300.0, len(ids)
        self.status["upload_error"] = f"HTTP {status}" if status else detail
        backoff = min(60.0, 2.0 ** min(self.status["failures"], 6))  # 1, 2, 4 ... 60 s
        self.status["failures"] += 1
        if status and retry_after.isdigit():
            return min(float(retry_after), 300.0), len(ids)
        return backoff * random.uniform(0.5, 1.0), len(ids)
