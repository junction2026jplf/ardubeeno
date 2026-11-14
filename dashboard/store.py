"""Durable receipt and aggregation. SQLite transactions are the ACK boundary."""
import json
import sqlite3
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import UTC, stamp


class Conflict(Exception):
    pass


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    method TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    UNIQUE(device_id, session_id, sequence)
                );
                CREATE INDEX IF NOT EXISTS event_history
                    ON events(device_id, source, occurred_at);
                CREATE TABLE IF NOT EXISTS device_receipts (
                    device_id TEXT PRIMARY KEY,
                    last_received_at TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def ingest(self, batch):
        received = stamp(datetime.now(UTC))
        inserted, duplicates, ids = 0, 0, []
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for event in batch.events:
                event_id = str(event.event_id)
                payload = json.dumps(event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
                existing = db.execute("SELECT device_id,payload FROM events WHERE event_id=?", (event_id,)).fetchone()
                if existing:
                    if existing["device_id"] != batch.device_id or existing["payload"] != payload:
                        raise Conflict("An event_id was already committed with different content")
                    duplicates += 1
                else:
                    try:
                        db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", (
                            event_id, batch.device_id, str(event.session_id), event.sequence,
                            stamp(event.occurred_at), event.source, event.method, event.kind, payload, received,
                        ))
                    except sqlite3.IntegrityError as exc:
                        raise Conflict("A device/session/sequence was already committed with another event_id") from exc
                    inserted += 1
                ids.append(event_id)
            db.execute("INSERT INTO device_receipts VALUES (?,?) ON CONFLICT(device_id) DO UPDATE SET last_received_at=excluded.last_received_at", (batch.device_id, received))
        return {"schema_version": 1, "device_id": batch.device_id, "committed_ids": ids,
                "inserted": inserted, "duplicates": duplicates, "received_at": received}

    def status(self, device_id, config):
        with self.connect() as db:
            latest = db.execute("SELECT max(occurred_at) AS occurred,max(received_at) AS received FROM events WHERE device_id=?", (device_id,)).fetchone()
            receipt = db.execute("SELECT last_received_at FROM device_receipts WHERE device_id=?", (device_id,)).fetchone()
            heartbeat = db.execute("SELECT payload FROM events WHERE device_id=? AND kind='heartbeat' ORDER BY occurred_at DESC,event_id DESC LIMIT 1", (device_id,)).fetchone()
            last_src = db.execute("SELECT source FROM events WHERE device_id=? ORDER BY occurred_at DESC,event_id DESC LIMIT 1", (device_id,)).fetchone()
            src_rows = db.execute("SELECT source,max(occurred_at) AS latest FROM events WHERE device_id=? AND kind IN ('bee_passage','coverage') GROUP BY source", (device_id,)).fetchall()
        heart = json.loads(heartbeat["payload"]) if heartbeat else None
        stale = not heart or (datetime.now(UTC) - datetime.fromisoformat(heart["occurred_at"])).total_seconds() > 120
        status = "never_seen" if not latest["received"] else "offline" if stale else heart["status"]
        return {"id": device_id, "name": config["name"], "timezone": config["timezone"], "status": status,
                "last_seen": latest["occurred"], "last_occurred_at": latest["occurred"],
                "last_received_at": receipt["last_received_at"] if receipt else latest["received"],
                "pending_events": heart["pending_events"] if heart else None,
                "active_source": heart["source"] if heart else (last_src["source"] if last_src else None),
                "sources": {r["source"]: r["latest"] for r in src_rows},
                "live_url": config.get("live_url"),
                "stale": stale}

    def history(self, device_id, start, end, zone, bucket, source, method="bee_tracks_v1"):
        # Load coverage across the complete device history: a healthy interval may
        # start before the query and its receipt may arrive much later.
        with self.connect() as db:
            rows = db.execute("SELECT kind,payload FROM events WHERE device_id=? AND source=? AND method=? AND (kind='coverage' OR (kind='bee_passage' AND occurred_at>=? AND occurred_at<?))", (device_id, source, method, stamp(start), stamp(end))).fetchall()
        passages, intervals = [], []
        for row in rows:
            item = json.loads(row["payload"])
            if row["kind"] == "coverage":
                a, b = datetime.fromisoformat(item["start_at"]), datetime.fromisoformat(item["end_at"])
                if a < end and b > start:
                    intervals.append((max(a, start), min(b, end)))
            else:
                passages.append((datetime.fromisoformat(item["occurred_at"]), item["assessment"]))
        intervals = union_intervals(intervals)
        passages.sort(key=lambda item: item[0])
        passage_times = [item[0] for item in passages]
        groups = []
        for a, b, label in boundaries(start, end, zone, bucket):
            counts = passages[bisect_left(passage_times, a):bisect_left(passage_times, b)]
            groups.append({"start": stamp(a), "end": stamp(b), "label": label,
                           **summarize(counts, intervals, a, b)})
        summary = summarize(passages, intervals, start, end)
        return {"summary": summary, "buckets": groups, "source": source, "timezone": zone,
                "bucket": bucket, "method": method, "schema_version": 1,
                "generated_at": stamp(datetime.now(UTC))}


def union_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def summarize(passages, intervals, start, end):
    assessments = [assessment for _, assessment in passages]
    duration = (end - start).total_seconds()
    recorded = sum(max(0, (min(b, end) - max(a, start)).total_seconds()) for a, b in intervals)
    starts = [a for a, _ in intervals]
    outside = 0
    for occurred, _ in passages:
        index = bisect_right(starts, occurred) - 1
        if index < 0 or occurred >= intervals[index][1]:
            outside += 1
    available = bool(recorded or assessments)
    return {
        "bees_observed": len(assessments) if available else None,
        "bees_with_detected_mites": assessments.count("mites_detected") if available else None,
        "bees_not_detected": assessments.count("not_detected") if available else None,
        "bees_assessment_uncertain": assessments.count("uncertain") if available else None,
        "recorded_seconds": recorded, "requested_seconds": duration,
        "events_outside_coverage": outside, "rate_available": bool(recorded and not outside),
        "availability": "recorded" if recorded else "events_only" if assessments else "unavailable",
        "coverage": "complete" if recorded == duration else "partial" if recorded else "missing",
    }


def boundaries(start, end, zone, bucket):
    tz = ZoneInfo(zone)
    local = start.astimezone(tz)
    if bucket == "5m":
        minute = (local.minute // 5) * 5
        cursor = local.replace(minute=minute, second=0, microsecond=0).astimezone(UTC)
        while cursor < end:
            following = cursor + timedelta(minutes=5)
            if following > start:
                yield max(cursor, start), min(following, end), cursor.astimezone(tz).strftime("%d %b %H:%M %z")
            cursor = following
    elif bucket == "hour":
        cursor = local.replace(minute=0, second=0, microsecond=0).astimezone(UTC)
        while cursor < end:
            following = cursor + timedelta(hours=1)
            yield max(cursor, start), min(following, end), cursor.astimezone(tz).strftime("%d %b %H:%M %z")
            cursor = following
    else:
        date = local.date()
        while True:
            cursor = datetime.combine(date, time.min, tz).astimezone(UTC)
            following = datetime.combine(date + timedelta(days=1), time.min, tz).astimezone(UTC)
            if cursor >= end:
                break
            yield max(cursor, start), min(following, end), date.isoformat()
            date += timedelta(days=1)
