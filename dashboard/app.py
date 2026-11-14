"""HTTP adapter for durable device uploads and the read-only dashboard."""
import csv
import hmac
import io
import json
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .models import Batch, UTC
from .store import Conflict, Store

STATIC = Path(__file__).parent / "static"
MAX_BODY_BYTES = 2 * 1024 * 1024


class BodyLimit:
    """Reject oversized uploads before JSON parsing, including chunked requests."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > MAX_BODY_BYTES:
                return await JSONResponse({"detail": "Request body exceeds 2 MiB"}, 413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(chunks)

        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)


def load_devices():
    try:
        devices = json.loads(os.environ.get("ARDUBEENO_DEVICES_JSON", "{}"))
        if not isinstance(devices, dict):
            raise ValueError
        for device_id, config in devices.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", device_id) or not isinstance(config, dict):
                raise ValueError
            if not isinstance(config.get("token"), str) or len(config["token"]) < 16:
                raise ValueError
            if not isinstance(config.get("name"), str) or not config["name"]:
                raise ValueError
            ZoneInfo(config["timezone"])
        return devices
    except (ValueError, TypeError, KeyError, ZoneInfoNotFoundError) as exc:
        raise ValueError("Invalid ARDUBEENO_DEVICES_JSON: each device needs a name, IANA timezone and token of at least 16 characters") from exc


def query_time(value):
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value):
            raise ValueError
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise HTTPException(422, "from/to must be RFC3339 timestamps with offsets") from exc


def create_app(db_path=None, devices=None):
    devices = load_devices() if devices is None else devices
    store = Store(db_path or os.environ.get("ARDUBEENO_DB", "dashboard/var/dashboard.sqlite3"))
    app = FastAPI(title="ArduBeeno receiving API", version="1.0.0")
    app.add_middleware(BodyLimit)
    app.state.store = store

    @app.exception_handler(sqlite3.Error)
    async def storage_failure(request, exc):
        # Do not expose database internals or acknowledge uncommitted events.
        return JSONResponse({"detail": "History storage unavailable; retry the same batch"}, 503,
                            headers={"Retry-After": "5"})

    def config_for(device_id):
        if device_id not in devices:
            raise HTTPException(404, "Unknown device")
        return devices[device_id]

    @app.get("/health")
    def health():
        with store.connect() as db:
            db.execute("SELECT 1").fetchone()
        return {"status": "ok", "schema_version": 1}

    @app.post("/api/v1/ingest/batch")
    def ingest(batch: Batch, authorization: str | None = Header(default=None)):
        token = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
        expected = devices.get(batch.device_id, {}).get("token", "")
        if not expected or not hmac.compare_digest(token.encode(), expected.encode()):
            raise HTTPException(401, "Invalid device credentials", headers={"WWW-Authenticate": "Bearer"})
        try:
            return store.ingest(batch)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/devices")
    def list_devices(response: Response):
        response.headers["Cache-Control"] = "no-store"
        return {"devices": [store.status(device_id, config) for device_id, config in devices.items()]}

    @app.get("/api/v1/devices/{device_id}/status")
    def status(device_id: str, response: Response):
        response.headers["Cache-Control"] = "no-store"
        return store.status(device_id, config_for(device_id))

    def history_result(device_id, start, end, tz, bucket, source, method):
        config = config_for(device_id)
        a, b = query_time(start), query_time(end)
        if not a < b or b - a > timedelta(days=366):
            raise HTTPException(422, "Query range must be positive and no longer than 366 days")
        zone = tz or config["timezone"]
        try:
            ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError):
            raise HTTPException(422, "Unknown IANA timezone")
        if bucket not in {"hour", "day"} or source not in {"field", "screen_demo", "replay"}:
            raise HTTPException(422, "Unsupported bucket or source")
        if method != "bee_tracks_v1":
            raise HTTPException(422, "Unsupported measurement method")
        result = store.history(device_id, a, b, zone, bucket, source, method)
        result["device"] = store.status(device_id, config)
        return result

    @app.get("/api/v1/devices/{device_id}/history")
    def history(device_id: str, response: Response, start: str = Query(alias="from"),
                end: str = Query(alias="to"), tz: str | None = None, bucket: str = "day",
                source: str = "field", method: str = "bee_tracks_v1"):
        response.headers["Cache-Control"] = "no-store"
        return history_result(device_id, start, end, tz, bucket, source, method)

    @app.get("/api/v1/devices/{device_id}/export.csv")
    def export(device_id: str, start: str = Query(alias="from"), end: str = Query(alias="to"),
               tz: str | None = None, bucket: str = "day", source: str = "field", method: str = "bee_tracks_v1"):
        result = history_result(device_id, start, end, tz, bucket, source, method)
        output = io.StringIO(newline="")
        columns = ["start", "end", "label", "bees_observed", "bees_with_detected_mites", "bees_assessment_uncertain",
                   "recorded_seconds", "requested_seconds", "events_outside_coverage", "rate_available", "availability", "coverage", "source", "method", "timezone", "device_id"]
        writer = csv.DictWriter(output, columns)
        writer.writeheader()
        for row in result["buckets"]:
            row = {**row, "source": source, "method": method, "timezone": result["timezone"], "device_id": device_id}
            # CSV must remain inert even if a configured identifier/method changes.
            writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for k, v in row.items()})
        return Response(output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="ardubeeno-{device_id}.csv"',
                                 "Cache-Control": "no-store"})

    @app.get("/", include_in_schema=False)
    def index():
        if not (STATIC / "index.html").exists():
            raise HTTPException(503, "Dashboard assets not installed")
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    # check_dir=False permits backend tests while the independent frontend is built.
    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
    return app
