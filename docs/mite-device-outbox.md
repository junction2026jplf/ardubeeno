# Arduino UNO Q API submission and local outbox handoff

This is the device implementation contract for the separate Arduino agent. **No device code is implemented or changed by this document.** The Python receiver and real dashboard live in `dashboard/`; the device must retain its own history even when that receiver is unreachable.

## API submission: start here

Implement the uploader in the UNO Q's Linux/Python application. The receiving endpoint already exists. Follow these steps to connect it; the sections below specify the full payload and local persistence contract.

### 1. Configure and expose the receiver

On the computer running the backend, follow [receiver setup](../dashboard/README.md), configure the real device and token, and start the server on a reachable interface:

```sh
export ARDUBEENO_DEVICES_JSON='{"uno-q-01":{"name":"Hive 01","timezone":"Europe/Helsinki","token":"REPLACE-WITH-A-PRIVATE-DEVICE-TOKEN"}}'
export ARDUBEENO_DB="$PWD/dashboard/var/receiver.sqlite3"
dashboard/.venv/bin/python -m dashboard --host 0.0.0.0 --port 8000
```

Replace the token placeholder on both sides with the same private token (at least 16 characters). On the Arduino, configure:

```sh
export ARDUBEENO_SERVER_URL='http://<receiver-computer-LAN-IP>:8000'
export ARDUBEENO_DEVICE_ID='uno-q-01'
export ARDUBEENO_DEVICE_TOKEN='<the-same-private-token>'
```

`ARDUBEENO_SERVER_URL` is the receiver's base URL without `/api/v1`. `0.0.0.0` is a bind address, not the destination to put on the Arduino. `localhost` on the Arduino addresses the Arduino itself. The separate demonstration service previously started on port 8001 was bound to loopback and configured for `demo-q`; configure a real receiver rather than treating that demo as an Arduino endpoint. Use HTTPS for a receiver outside the trusted local link.

Completion check: from the board, `GET $ARDUBEENO_SERVER_URL/health` returns HTTP 200 and `{"status":"ok","schema_version":1}`. This checks connectivity; the authenticated POST below checks the device credentials.

### 2. Prepare durable readings

Use one `bee_passage` event per completed tracked bee, including bees without detected mites. `assessment` determines whether that event also counts as a bee with mites. The API derives totals from these events; it does not accept cumulative `bees_observed` or `mites_count` readings as event fields. Record healthy `coverage` intervals and `heartbeat` events as specified under [Receiver wire format](#receiver-wire-format).

Allocate IDs and the session sequence once, and commit the event plus outbox row locally before uploading. A batch contains `schema_version: 1`, the configured `device_id`, and 1–500 events. Keep its encoded JSON body within 2 MiB; start with batches of 100. The complete three-kind JSON example below can be saved as `batch.json` for a protocol smoke test. Its fixed IDs and historical timestamps are examples, not IDs to reuse for live readings.

### 3. Submit the pending batch

Request: `POST $ARDUBEENO_SERVER_URL/api/v1/ingest/batch`, with `Authorization: Bearer <device-token>` and `Content-Type: application/json`.

This standard-library Python example sends an **already persisted** `batch.json`. It deliberately does not generate fresh IDs, rewrite timestamps, retry automatically, or modify the local database:

```python
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

batch_bytes = Path("batch.json").read_bytes()
batch = json.loads(batch_bytes)
device_id = os.environ["ARDUBEENO_DEVICE_ID"]
if batch["device_id"] != device_id:
    raise ValueError("Pending batch belongs to another device")

request = Request(
    os.environ["ARDUBEENO_SERVER_URL"].rstrip("/") + "/api/v1/ingest/batch",
    data=batch_bytes,
    method="POST",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["ARDUBEENO_DEVICE_TOKEN"],
    },
)
try:
    with urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError("Unexpected response; retain this batch")
        ack = json.load(response)
except HTTPError as error:
    # Apply the status-specific rules below; the local batch stays pending.
    raise RuntimeError(f"Upload rejected with HTTP {error.code}; retain this batch") from None

sent_ids = {event["event_id"] for event in batch["events"]}
committed = ack.get("committed_ids")
if (ack.get("schema_version") != 1 or ack.get("device_id") != device_id
        or not isinstance(committed, list)
        or any(not isinstance(event_id, str) for event_id in committed)
        or set(committed) != sent_ids):
    raise RuntimeError("Invalid acknowledgement; retain this batch")

# Integration point: atomically mark only these IDs as ACKed in the local outbox.
# Keep the original local event/history rows.
print(f"Receiver committed {len(sent_ids)} event(s)")
```

For the worker implementation, catch connection errors, timeouts and invalid JSON as retryable failures while preserving the exact queued events. Read [Acknowledgement and retries](#acknowledgement-and-retries) before adding retry logic. The response acknowledges **events**, including coverage and heartbeats, so `inserted` is not a bee counter.

### 4. Verify the received readings

Query `GET /api/v1/devices/uno-q-01/history` with `from`, `to`, `source`, `bucket` and optionally `tz`. Example parameters for the historical batch below:

```text
from=2026-11-14T08:00:00Z
to=2026-11-14T08:01:00Z
source=screen_demo
bucket=hour
tz=Europe/Helsinki
```

On a fresh receiver database, that sample produces `summary.bees_observed=1`, `summary.bees_with_detected_mites=1`, and `summary.recorded_seconds=30`. Submit the identical batch again: all three event IDs remain acknowledged, `inserted=0`, `duplicates=3`, and the bee counters stay at one. Add a new `bee_passage` with `assessment=not_detected` and a fresh persisted ID/sequence: only the overall observed-bee counter increases.

The dashboard must use the matching source and date range. Its default recent range will not necessarily include the example's fixed historical date. `GET /api/v1/devices/uno-q-01/status` distinguishes server receipt time from old observation time. Interactive schemas are at `/docs` on the same receiver.

### 5. Finish the device integration

Completion means that local history survives a restart, an offline batch uploads after reconnect, an identical replay leaves server counts unchanged, and ACKed events remain available in local history. Implement the transactional queue using [Durable local database](#durable-local-database), then run the [Device completion checks](#device-completion-checks). No changes to the receiver wire schema are needed for this integration.

## Boundary and measurement

Run the event store and uploader on the UNO Q's Linux side. Keep the STM32 matrix and existing Bridge calls separate. The receiving server can run on a laptop, another computer on the LAN, or a server reachable over HTTPS. It does not open the camera or run inference. Local history must keep working while it is unreachable.

The existing shared `applab/ardubeeno/python/monitor.py` counts a clean→positive temporal-consensus transition in a centre crop. That is a positive episode, not all observed bees and not reliable unique infested bees. Do not convert its cumulative `/state.count` into the new counters. Both `python/main.py` (App Lab) and `python/serve.py` (standalone runner) call this shared monitor, so integrate a single local event/history service beneath those runners.

The new measurement comes from completed **bee tracks**:

- Finalize every eligible bee passage once when its track exits the observation zone or expires. Count it in `bees_observed` regardless of mite verdict.
- Assign `mites_detected`, `not_detected`, or `uncertain` from the track's aggregated mite assessment. Only `mites_detected` increments `bees_with_detected_mites`.
- Preserve uncertain assessments in the observed total; do not call them mite-free.
- One track with several mite boxes still contributes one observed bee and one bee with detected mites. Repeated frames do not add bee counts.
- A later visit by the same physical bee may create another passage. This is not a census of unique colony members or an estimate of total colony mites.
- Start a new session UUID on process startup. Close unfinished tracks as interrupted locally; exclude them from finalized passage events. Do not manufacture completed passages after a crash. Persist a unique local `(session_id, track_id)`→event identity mapping to prevent repeated finalization.

## Durable local database

Use a SQLite database on a persistent Linux filesystem, not an image/container layer or transient app checkout. Make its path configurable. The App Lab volume mapping and permissions must be verified on the actual board across both app redeployment and reboot; this document does not assume a particular host path is mounted inside App Lab.

Recommended tables (device-side design, not the receiver's schema):

| Table | Required responsibilities |
| --- | --- |
| `sessions` | Session UUID, boot/start monotonic reference, raw wall-clock reference, clock quality, source, model/config revisions, next sequence. |
| `local_events` | Stable UUID, session UUID, sequence, kind, raw monotonic times, raw wall times, source, assessment/status, optional local track ID, model/config revision, eventual normalized UTC fields and sealed wire JSON. Keep this as local history. |
| `outbox` | Event UUID foreign key, state (`blocked_time`, `pending`, `leased`, `acked`, `quarantined`), attempt count, next retry UTC/monotonic schedule, lease expiry, last error and acknowledgement time. Unique event UUID. |
| `track_finalizations` | Unique `(session_id, track_id)` and event UUID, written with the event; duplicate finalization reuses the existing event. |
| `clock_anchors` | Session/boot monotonic offset→trusted UTC mapping and uncertainty; retain the original raw clock evidence. |

For each passage, coverage interval, or heartbeat:

1. Begin one SQLite transaction and allocate the next session sequence, starting at 1. The sequence is shared by **all event kinds**, not one sequence per kind.
2. Insert the local event with its stable UUID; for a passage also insert its unique track-finalization mapping.
3. Insert its outbox row in the **same transaction**. Use `pending` only when the wire timestamp is trusted and the wire body has been sealed; otherwise use `blocked_time`.
4. Commit. Only now report that the event has been recorded durably. Display totals should derive from committed passage events, not an independent resettable counter.

If the process crashes before commit, neither the local event nor its outbox row exists. If it crashes after commit, the uploader will find the pending event on restart. Update acknowledgements in a separate transaction only after receiving the server response.

Use short transactions, a bounded SQLite busy timeout, one writer, and foreign keys. WAL mode is suitable; prefer synchronous durability appropriate for abrupt power loss (e.g. `synchronous=FULL`) and benchmark on the board. Keep database operations and network requests outside the monitor condition lock. Do not hold a transaction open across network I/O. Use the SQLite backup API for consistent backups; do not copy only the database file while WAL writes are active.

An uploader worker can lease a bounded batch, release the database lock, make the HTTP request, and then acknowledge IDs in a new transaction. Leases must expire after a process crash. A single uploader is sufficient initially, but the schema should prevent two workers from owning the same unexpired lease.

## Receiver wire format

Endpoint: `POST /api/v1/ingest/batch` with `Content-Type: application/json` and `Authorization: Bearer <per-device token>`.

The body has exactly `schema_version`, `device_id`, and `events`. Use schema version 1. Send 1–500 events per batch; start with at most 100 for predictable latency on the board. The server rejects unknown event fields, so keep local track IDs, monotonic timestamps, image paths, and model metadata in local history until a versioned receiver extension supports them.

Every wire event contains:

| Field | Value |
| --- | --- |
| `event_id` | Stable UUID generated once when the local event is created. |
| `session_id` | UUID for the process session that captured the event; retain it when replaying old sessions. |
| `sequence` | Integer from 1 through 9223372036854775807, unique for this device/session across all event kinds. |
| `kind` | `bee_passage`, `coverage`, or `heartbeat`. |
| `occurred_at` | RFC3339 timestamp with explicit UTC offset; serialize UTC as `Z`. This is capture/completion time, never retry/upload time. |
| `source` | `field`, `screen_demo`, or `replay`. Never relabel demonstrations as field observations. |
| `method` | Exactly `bee_tracks_v1` in schema version 1 (also the default). Use the same agreed method for passages and their coverage. A different counting method requires a receiver/schema extension. |

Kind-specific fields:

| Kind | Additional fields |
| --- | --- |
| `bee_passage` | `assessment`: `mites_detected`, `not_detected`, or `uncertain`. One event is one completed bee passage. |
| `coverage` | `start_at`, `end_at`, both offset-aware; `start_at < end_at` and `occurred_at == end_at`. Coverage is a **healthy recorded interval**; maximum span 366 days. |
| `heartbeat` | `status`: `recording`, `idle`, `camera_error`, or `storage_error`; optional `pending_events` integer ≥0. |

Example batch, showing all three event kinds:

```json
{
  "schema_version": 1,
  "device_id": "uno-q-01",
  "events": [
    {
      "event_id": "fa03a447-60b0-4ce3-b164-482ae62073fb",
      "session_id": "bce69a3f-aac4-4b91-96f2-d0f1d8748c40",
      "sequence": 1,
      "kind": "bee_passage",
      "occurred_at": "2026-11-14T08:00:10Z",
      "source": "screen_demo",
      "method": "bee_tracks_v1",
      "assessment": "mites_detected"
    },
    {
      "event_id": "81877265-70ad-4893-ad85-c1975323a25e",
      "session_id": "bce69a3f-aac4-4b91-96f2-d0f1d8748c40",
      "sequence": 2,
      "kind": "coverage",
      "occurred_at": "2026-11-14T08:00:30Z",
      "source": "screen_demo",
      "method": "bee_tracks_v1",
      "start_at": "2026-11-14T08:00:00Z",
      "end_at": "2026-11-14T08:00:30Z"
    },
    {
      "event_id": "d54859f1-a624-4ae2-9ac5-727a76bcbf7c",
      "session_id": "bce69a3f-aac4-4b91-96f2-d0f1d8748c40",
      "sequence": 3,
      "kind": "heartbeat",
      "occurred_at": "2026-11-14T08:00:30Z",
      "source": "screen_demo",
      "method": "bee_tracks_v1",
      "status": "recording",
      "pending_events": 3
    }
  ]
}
```

Replay these exact IDs only for these exact events. A new real passage needs a new UUID and sequence. Store sealed event JSON locally so retries cannot change field values, timestamps, source, assessment, or method. Byte-identical JSON serialization on every retry is a simple safe implementation; the receiver compares normalized immutable event content rather than relying on whitespace.

Example terminal request once the device agent has written a real pending batch to `batch.json` and provisioned its environment:

```sh
curl --fail-with-body \
  --connect-timeout 5 --max-time 20 \
  --request POST "$ARDUBEENO_SERVER_URL/api/v1/ingest/batch" \
  --header "Authorization: Bearer $ARDUBEENO_DEVICE_TOKEN" \
  --header 'Content-Type: application/json' \
  --data-binary @batch.json
```

The uploader should use an HTTP client with explicit connect/read timeouts, not launch `curl` for each batch. Avoid verbose HTTP logging that exposes the Authorization header. The example expands a token in a process argument; use the application's HTTP client for operational uploading and do not paste real credentials into shell history or shared logs.

## Acknowledgement and retries

The server responds only after committing the entire batch:

```json
{
  "schema_version": 1,
  "device_id": "uno-q-01",
  "committed_ids": [
    "fa03a447-60b0-4ce3-b164-482ae62073fb",
    "81877265-70ad-4893-ad85-c1975323a25e",
    "d54859f1-a624-4ae2-9ac5-727a76bcbf7c"
  ],
  "inserted": 3,
  "duplicates": 0,
  "received_at": "2026-11-14T08:00:31Z"
}
```

An exact replay is successful too: `inserted: 0`, `duplicates: 3`, with all three IDs acknowledged. A lost response is therefore safe to retry. The response's `received_at` is a server receipt time, not a replacement capture time.

Only acknowledge event IDs that were in the transmitted batch and appear in a well-formed successful response for the same device and schema version. Do not mark the entire outbox delivered from a sequence high-water mark; offline batches can arrive out of order. The receiver accepts old, previously unseen events and sequence gaps.

| Outcome | Device action |
| --- | --- |
| Successful ACK | Mark exactly the acknowledged outbox rows `acked` atomically. Keep the local history rows. Reset transient retry backoff. |
| Timeout, disconnected network, invalid/truncated response | Keep the same events pending and retry the same identities. The server might already have committed. |
| HTTP 503 or other transient 5xx | Keep pending; exponential backoff with jitter, for example 1, 2, 4, 8… up to 60 seconds. Honor a valid `Retry-After` when returned. |
| HTTP 401 | Stop aggressive retries; retain data and expose a credential/configuration fault. Correct device ID/token configuration, then retry unchanged events. |
| HTTP 413 | Request exceeded the 2 MiB body limit. Split into smaller batches while keeping event identities and payloads unchanged; acknowledge each smaller batch independently. |
| HTTP 422 | Entire batch rejected: retain all rows. Quarantine the validation fault for operator/developer attention. Isolate an invalid event if necessary without changing already sealed events or inventing replacement identities. |
| HTTP 409 | Entire batch rejected: event UUID was reused with different content, or the session sequence was reused for another event. Quarantine the conflict and investigate local identity allocation. Never bypass the fault by generating a new ID for the same passage. |

Authenticate with the configured device ID's own token. Receiver configuration is `ARDUBEENO_DEVICES_JSON`, a mapping such as `{"uno-q-01":{"name":"Hive 01","timezone":"Europe/Helsinki","token":"<provisioned secret>"}}`. Unknown devices are not auto-registered. Provision tokens outside source control, use a unique token per board, and rotate them through configuration. Use HTTPS for remote transport; plain HTTP is acceptable only on the explicitly trusted local demo link. The receiver's read API is intended for a trusted local/private deployment unless protected by an authenticated gateway.

The receiver rejects timestamps more than five minutes in its future. It accepts historical offline uploads. Do not rewrite an old capture timestamp as “now” to make a rejected event pass.

## Coverage and clock handling

Coverage is not a heartbeat and is not inferred from Wi-Fi connectivity. Emit healthy intervals only while camera capture, bee detection/tracking, per-bee assessment, and local durable recording are working. Close/persist coverage intervals periodically (e.g. every 30 seconds) and before a clean shutdown or known recording fault. Publish status separately as heartbeat events. Overlapping coverage intervals are permitted; the receiver unions them before computing recorded duration, so retries or overlap cannot inflate coverage.

After an unexpected crash, only already committed intervals are recorded coverage. Do not bridge the gap between the last good interval and the next process startup. The dashboard can retain a real passage that arrived without corresponding coverage while warning that its rate denominator is incomplete; never invent a full-day denominator to hide that gap.

Keep capture monotonic times and raw wall-clock evidence locally. A correct clock may remain usable offline; loss of internet alone does not invalidate it. If the wall clock is untrusted after boot, store history and `blocked_time` outbox entries but do not transmit guessed calendar timestamps. Once the same boot/session obtains a trusted UTC anchor, map its monotonic offsets to UTC, seal the wire event and release its outbox row in one transaction. Retain the anchor and original clock evidence locally. If no trustworthy anchor spans a reboot, retain that session for review rather than inventing dates. An already sealed/sent event must never have its timestamp modified under the same UUID.

Use event capture/completion times to reconstruct history; the server receipt time only tells whether communication is recent. A fresh connection uploading an old backlog does not prove that the camera is recording now. Send fresh heartbeat events alongside backlog batches (with their own new sequences), allowing the receiver to distinguish old capture activity from recent receipt activity.

## Retention, capacity, and recovery

- An ACK means the receiver has a committed copy; it is not permission to erase the board's local history. Apply a separate, explicit local retention policy, initially retaining all small event records unless configured otherwise.
- Never silently prune unacknowledged events. Bound optional images first; images are not part of the v1 event protocol.
- Track local database size, free space, pending/blocked/quarantined counts, oldest pending timestamp, and last successful ACK. Warn before the storage reserve is exhausted. Reserve enough space to commit fault/status records when possible.
- If local writes fail, stop claiming durable coverage, expose a storage fault in the local UI/matrix as appropriate, and keep any uncommitted-event count visible. Network upload failure alone must not stop capture/history while local storage remains healthy.
- A local “reset count” changes a display baseline marker only. It does not reset event sequences, delete history, or clear unsent outbox rows. The existing `Monitor.reset_count()` behavior must be adapted deliberately by the device agent.
- Receiver recovery from backup can request replay: ACKed local rows can be re-enqueued with the same IDs and sealed payloads. Document the receiver's retention/backup policy separately from the board's.

## Device completion checks

1. Run a known video with no-mite, mite-positive, uncertain, and overlapping bee tracks. Reconcile both counts against finalized track IDs; a positive track must be a subset of all tracks.
2. Record with the receiver unavailable; restart the app and reboot the board. Confirm history and pending identities survive in the verified persistent volume.
3. Restore connectivity and resend the same batch repeatedly, including a simulated lost ACK. Confirm server counts change once and local history remains after acknowledgement.
4. Crash between local commit and upload, and between server commit and local ACK. Expired leases must recover without duplicate passages.
5. Send batches from multiple old sessions out of order. Confirm history uses capture times and device status does not mistake an old heartbeat for a current recording state.
6. Exercise clock-untrusted boot, clock correction, camera failure, storage failure, bad token, 409 conflict, and 422 validation. Confirm no gap becomes a measured zero and no fault silently drops events.
7. Verify local reset, App Lab redeploy, backup/restore, and model/source changes. Keep `screen_demo`, `replay`, and `field` histories distinct.

The receiver's `/docs` OpenAPI page is the executable field reference. If the implementation evolves, version the payload explicitly and update this handoff before changing the Arduino uploader.
