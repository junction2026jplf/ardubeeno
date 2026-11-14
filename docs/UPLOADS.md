# Uploads: from the board to the dashboard

The board records what it sees in a local SQLite history and uploads it to the receiver in `dashboard/`. The board
keeps working and keeps its history while the receiver is unreachable, and sends the backlog later.

- Board side: `applab/ardubeeno/python/outbox.py` (`Recorder`), fed by `monitor.py` in both runners (`main.py` in
  App Lab, `serve.py` without it). Standard library only.
- Receiver side: `dashboard/` ([dashboard/README.md](../dashboard/README.md)). Its `/docs` page is the OpenAPI
  reference for every field.

## Setup

On the laptop, two .gitignored files with the same token (at least 16 characters):

```sh
# demo/receiver.env
ARDUBEENO_DEVICES_JSON='{"uno-q-01":{"name":"Hive 01","timezone":"Europe/Helsinki","token":"<long random token>"}}'
ARDUBEENO_DB=dashboard/var/receiver.sqlite3

# demo/ardubeeno.env
ARDUBEENO_SERVER_URL=http://<laptop-ip>:8000
ARDUBEENO_DEVICE_ID=uno-q-01
ARDUBEENO_DEVICE_TOKEN=<the same token>
ARDUBEENO_SOURCE=screen_demo
```

```sh
demo/start-receiver.sh                                    # receiver + dashboard on port 8000, all interfaces
make applab-config BOARD=arduino@ardu-bee-no.local        # copy demo/ardubeeno.env to the board, check /health from there
```

`ARDUBEENO_SERVER_URL` is the receiver's base URL as seen **from the board** (not `localhost`, not `0.0.0.0`). Plain
HTTP is fine on a trusted local link; use HTTPS for anything else.

Board settings, from the environment or `KEY=value` lines in `data/ardubeeno.env` in the app folder:

| Setting | Meaning |
|---|---|
| `ARDUBEENO_SERVER_URL`, `ARDUBEENO_DEVICE_ID`, `ARDUBEENO_DEVICE_TOKEN` | All three turn uploads on. With none set, the app runs without uploads; with only some, it prints `upload off: ...`. |
| `ARDUBEENO_SOURCE` | `field` (default), `screen_demo` or `replay`. The dashboard keeps them apart. `serve.py --source` / `SOURCE` in App Lab override it. |
| `ARDUBEENO_OUTBOX_DB` | History database, default `data/outbox.sqlite3` in the app folder. |
| `ARDUBEENO_CLOCK_TRUSTED` | `1` / `0` overrides the NTP check (see Clock). |

Events are recorded only in multi-bee mode; the single-bee mode has no bee tracks, so it records nothing.

## What is recorded

| Kind | When | Wire fields besides the common ones |
|---|---|---|
| `bee_passage` | A tracked bee ends (left the frame, lost, or a scene cut). One event per track, also for clean and uncertain bees. | `assessment`: `mites_detected`, `not_detected` or `uncertain` |
| `coverage` | Time the pipeline was actually recording: closed every 15 s, before each passage, on a frame gap over 5 s and at shutdown. | `start_at`, `end_at` (`occurred_at` = `end_at`) |
| `heartbeat` | Shortly after start, then every 20 s, on reconnect, and `idle` at shutdown. | `status`: `recording`, `idle`, `camera_error` or `storage_error`; `pending_events` |

Common fields: `event_id` (UUID, made once), `session_id` (UUID per app start), `sequence` (from 1, shared by all
kinds in a session), `kind`, `occurred_at` (RFC3339 UTC: the capture time, never the upload time), `source`,
`method` (always `bee_tracks_v1`).

Assessment of a bee: `mites_detected` after `min_hits` positive checks (2) or one strong check; `not_detected` only
after at least 2 checks with no positive one; everything else (one check, mixed results, never checked) is
`uncertain`. Only `mites_detected` counts as a bee with mites; uncertain bees still count as observed.

A passage is one visit, not one bee: a bee that leaves and comes back counts again.

Locally each passage also keeps its track ID and check details (`checks`, `positives`, `strong`, `score`,
`min_hits`, and `pre_reset` / `interrupted`, see below). The receiver rejects unknown fields, so these stay on the
board.

## The local database

One SQLite file (WAL, `synchronous=FULL`, foreign keys). A writer thread commits events; an uploader thread sends
them. The camera loop only puts events on an in-memory queue and never waits for the database or the network.

| Table | Holds |
|---|---|
| `sessions` | One row per app start: boot ID, start times, clock trust, source, model/settings revision, next sequence |
| `local_events` | Every event with its monotonic and wall times, value, track ID, local details and, once sealed, the exact wire JSON |
| `outbox` | Upload state per event: `blocked_time`, `pending`, `leased`, `acked`, `quarantined`; attempts, lease, last error |
| `track_finalizations` | `(session_id, track_id)` → event: a track becomes a passage at most once |
| `clock_anchors` | Monotonic → UTC mappings with their evidence |
| `local_meta` | The display baseline for resets |

Each event, its track mapping and its outbox row are written in one transaction, so after a crash an event either
exists completely or not at all. Acknowledged events stay in the history; nothing is deleted.

## Upload and acknowledgement

`POST {ARDUBEENO_SERVER_URL}/api/v1/ingest/batch` with `Authorization: Bearer <token>` and a body of
`{"schema_version": 1, "device_id": ..., "events": [...]}`: up to 100 events per batch (the receiver accepts 500),
at most 2 MiB. The uploader leases the oldest pending events (this session's heartbeats first, so the
dashboard sees a fresh status before an old backlog), sends them, and marks exactly the acknowledged IDs `acked`. A
lease expires after 120 s, so events of a crashed upload are sent again.

The receiver answers only after committing the whole batch. A replay of identical events is a success too
(`inserted: 0`, `duplicates: n`), so resending after a lost response is safe.

| Response | Board |
|---|---|
| 200 with `committed_ids` equal to the sent IDs | Mark them `acked`, reset the backoff |
| 200 with anything else, timeout, connection error | Keep them pending, retry the same events |
| 5xx | Retry with backoff 1, 2, 4 ... 60 s with jitter, or `Retry-After` (at most 300 s) |
| 401, other 4xx | Credentials fault (shown on the page), retry after 300 s |
| 409, 413, 422 with several events | Halve the batch to find the bad event, same IDs and content |
| 409, 413, 422 with one event | Quarantine that event, keep it, go on with the rest |

409 means an event ID or session sequence was reused with different content; 422 a validation error (e.g. a
timestamp more than 5 minutes in the receiver's future). Events are never changed or given a new ID to get past
either.

## Clock

The board checks with `adjtimex()` whether the clock is NTP-synchronised. If it is, events get their UTC time and
wire JSON ("sealed") when they are recorded. If not, they are recorded as `blocked_time` and not sent. When the clock
becomes trusted later in the same boot, the waiting events of this boot are sealed with the new monotonic → UTC
anchor. Events of an earlier boot without a trusted clock stay blocked: their dates are unknown. A sealed event's
time never changes.

## Reset, restart and shutdown

- **Reset** (button on the page, or `/reset`): sets the page's and LED matrix's counts to zero. It stores a baseline
  (the last event's row) in `local_meta` and changes nothing else: no history deleted, sequences go on, unsent
  events stay queued. Bees in view at the reset are marked `pre_reset`; they still become passages (and are
  uploaded) when they leave, but don't count towards the new totals.
- **Restart**: the counts since the last reset (per source) are rebuilt from the passages after the baseline,
  without the `pre_reset` ones, so the page and LED show the same numbers as before.
- **Shutdown**: pending mite checks get 0.3 s to finish; bees still in view become passages marked `interrupted`;
  the open coverage is closed; then the board waits up to 3 s for the queue to be uploaded and sends an `idle`
  heartbeat. Whatever isn't acknowledged by then is sent on the next start.
- **Storage failure**: if a write fails, the page shows a storage fault, coverage stops (unrecorded time is not
  reported as recorded) and the heartbeat says `storage_error`. A network failure alone never stops recording.

## Check it

```sh
dashboard/.venv/bin/python -m unittest dashboard.tests.test_api        # receiver: auth, replay, conflicts, coverage, CSV
```

On the board, with the dashboard open on `source=screen_demo`:

1. Show bees: passages appear within a few seconds, and the page's upload status shows the last acknowledgement.
2. Stop the receiver, show more bees, start it again: the queued events arrive, counts change once.
3. Restart the app: the page shows the same counts as before; reset sets them to zero.
4. A wrong token in `demo/ardubeeno.env`: the page shows the credentials fault, nothing is lost.
