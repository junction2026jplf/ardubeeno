"""Explicit synthetic history for a separate demo DB, never automatic startup data."""
import argparse
import json
import os
import shlex
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from .models import Batch, UTC, stamp
from .store import Store

DEMO_DEVICES = {"demo-q": {"name": "UNO Q · Demonstration", "timezone": "Europe/Helsinki",
                            "token": "demo-only-not-a-production-secret"}}


def seed(path, days=7):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally mixing fixtures into real history.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    store = Store(path)
    session = str(uuid4())
    sequence = 0
    pending = []
    now = datetime.now(UTC)
    end = now.replace(minute=0, second=0, microsecond=0)
    begin = end - timedelta(days=days)
    total = 0

    def emit(kind, occurred, **fields):
        nonlocal sequence, total
        sequence += 1
        pending.append({"event_id": str(uuid4()), "session_id": session, "sequence": sequence,
                        "occurred_at": stamp(occurred), "source": "screen_demo", "kind": kind, **fields})
        total += kind == "bee_passage"
        if len(pending) == 500:
            flush()

    def flush():
        if pending:
            batch = Batch.model_validate({"schema_version": 1, "device_id": "demo-q", "events": pending})
            store.ingest(batch)
            pending.clear()

    for hour in range(days * 24):
        # Leave explicit gaps rather than implying zero bees during a camera outage.
        if hour % 43 in {11, 12, 13}:
            continue
        start = begin + timedelta(hours=hour)
        finish = start + timedelta(hours=1)
        emit("coverage", finish, start_at=stamp(start), end_at=stamp(finish))
        count = 12 + (hour * 17) % 29
        for bee in range(count):
            occurred = start + timedelta(seconds=(bee + 1) * 3600 / (count + 1))
            assessment = "mites_detected" if (bee + hour) % 19 == 0 else "uncertain" if (bee + hour) % 31 == 0 else "not_detected"
            emit("bee_passage", occurred, assessment=assessment)
    emit("heartbeat", now, status="recording", pending_events=0)
    flush()
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="dashboard/var/demo.sqlite3", help="New separate database; refuses an existing file")
    parser.add_argument("--days", type=int, choices=range(1, 31), default=7, metavar="1..30")
    args = parser.parse_args()
    try:
        total = seed(args.db, args.days)
    except FileExistsError:
        parser.exit(2, "Demo database already exists; choose a new --db path. Existing history was not changed.\n")
    print(f"Created {total:,} synthetic bee passages in {args.db}. Select Screen demo in the dashboard.")
    print("Run from the repository root:")
    print("ARDUBEENO_DB=" + shlex.quote(args.db) + " ARDUBEENO_DEVICES_JSON=" +
          shlex.quote(json.dumps(DEMO_DEVICES)) + " dashboard/.venv/bin/python -m dashboard")


if __name__ == "__main__":
    main()
