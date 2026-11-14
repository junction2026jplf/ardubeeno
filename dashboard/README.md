# ArduBeeno receiver and dashboard

This is the receiving application: a Python API, persistent server-side SQLite history, and a real API-connected dashboard. The Arduino's local SQLite history and outbox are **specified, not implemented here**. The device agent's handoff is [the outbox contract](../docs/mite-device-outbox.md).

The dashboard shows all observed bee passages and the subset with detected mites in one stacked chart. The red segment is included in the total, never added on top of the total. Solo controls isolate either metric. Counts, coverage, source, time range, and CSV use the same server data. Uncertain assessments stay in the overall observed count.

## Install and run

Run from the repository root with Python 3.12 or later. This environment is separate from model training:

```sh
uv venv --python 3.12 dashboard/.venv
uv pip install --python dashboard/.venv/bin/python -r dashboard/requirements.txt
```

Configure a device with a private token in your shell or service environment. The following is a configuration template; replace the placeholder before sending device data:

```sh
export ARDUBEENO_DEVICES_JSON='{"uno-q-01":{"name":"Hive 01","timezone":"Europe/Helsinki","token":"REPLACE-WITH-A-LONG-RANDOM-DEVICE-TOKEN"}}'
export ARDUBEENO_DB="$PWD/dashboard/var/receiver.sqlite3"
dashboard/.venv/bin/python -m dashboard
```

Open <http://127.0.0.1:8000>. With no received events, the dashboard shows an empty state. It never substitutes demonstration data for missing device data. API schemas are available at `/docs`.

For a populated demonstration without an Arduino, explicitly create a separate demo database:

```sh
dashboard/.venv/bin/python -m dashboard.demo
```

Run the exact launch command it prints, open the dashboard, and choose **Screen demonstration** as the source. The seeder refuses to overwrite an existing database; use `--db dashboard/var/another-demo.sqlite3` for a new one. Its public demo credential is for synthetic data only. Production starts never seed events automatically.

Keep the device token out of frontend code and source control. The sender must use the same configured token. For multiple boards, add another device entry with its own token and ID. The configured device timezone controls the dashboard's local calendar boundaries.

The default address is loopback. To receive from an Arduino on your trusted local network, bind to your computer's LAN interface or `0.0.0.0`:

```sh
dashboard/.venv/bin/python -m dashboard --host 0.0.0.0 --port 8000
```

Then use `http://<computer-LAN-IP>:8000/api/v1/ingest/batch` on the Arduino. Do not use `localhost` on the board to address this receiver. Ingestion requires a device bearer token; dashboard reads are intended for a trusted local network. Before exposing the receiver to the internet, place it behind HTTPS and authenticated access for dashboard/read routes. Hosting a static webpage alone does not deploy this Python API.

## What is stored

The receiver persists immutable bee-passage, recording-coverage and heartbeat events. It acknowledges an entire accepted batch only after commit. Replaying identical event identities is safe; changing an already stored event or reusing a session sequence for another event rejects the batch. Out-of-order delivery after an offline period is supported. Event time places observations on the historical chart; receipt time indicates when the receiver last heard from a board.

Healthy coverage intervals are merged before aggregation so overlaps and retries cannot inflate recorded duration. Counts with no coverage are identified as coverage unknown; missing data is not interpreted as a zero-bee observation. All finalized bee tracks must be sent, including tracks with no detected mites and uncertain assessments. The existing detector's positive-transition counter cannot provide the total observed-bee denominator.

Keep `ARDUBEENO_DB` on durable local storage. Do not delete the database when restarting or redeploying. Back up a running SQLite database with its backup API rather than copying just the main file while WAL writes are active. The server is separate from the detector and does not open a camera, change models, or reset the Arduino matrix.

## Device integration

See [the device outbox handoff](../docs/mite-device-outbox.md) for exact event JSON, acknowledgements, retry/error handling and on-board transaction design. The onboard event and outbox row must commit together; acknowledgement changes delivery state without deleting local history. No SSH access or device deployment is required to run this receiver.

The earlier [architecture draft](../docs/mite-dashboard-draft/architecture.md) documents design exploration. This README, the outbox handoff and the implemented OpenAPI contract describe the receiving application.

## Checks

```sh
uv pip install --python dashboard/.venv/bin/python -r dashboard/requirements-dev.txt
dashboard/.venv/bin/python -m unittest discover -s dashboard/tests -v
node --check dashboard/static/app.js
```

The API tests cover authentication and validation, replay and persistence, whole-batch conflict rollback, offline/out-of-order uploads, source separation, coverage union and gaps, invalid rate denominators, fractional timestamp boundaries, daylight-saving days and repeated hours, body limits, and CSV parity. Tests use temporary databases. The frontend uses no build step or CDN dependencies.
