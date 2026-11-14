"""Behavior checks for the upload/ack boundary and historical interpretation."""
import csv
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from dashboard.app import create_app


class ReceivingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "events.sqlite3"
        self.devices = {"uno-q-01": {"token": "a-valid-device-secret", "name": "Hive 01", "timezone": "Europe/Helsinki"}}
        self.client = TestClient(create_app(self.path, self.devices))
        self.addCleanup(self.client.close)
        self.session = str(uuid4())
        self.sequence = 0

    def event(self, kind="bee_passage", **changes):
        self.sequence += 1
        item = {"event_id": str(uuid4()), "session_id": self.session, "sequence": self.sequence,
                "occurred_at": "2026-11-13T10:30:00Z", "source": "field", "kind": kind}
        if kind == "bee_passage":
            item["assessment"] = "not_detected"
        elif kind == "coverage":
            item.update(start_at="2026-11-13T10:00:00Z", end_at="2026-11-13T11:00:00Z", occurred_at="2026-11-13T11:00:00Z")
        else:
            item["status"] = "recording"
        return {**item, **changes}

    def upload(self, *events, token="a-valid-device-secret", client=None):
        return (client or self.client).post("/api/v1/ingest/batch", json={"schema_version": 1,
                    "device_id": "uno-q-01", "events": list(events)}, headers={"Authorization": "Bearer " + token})

    def history(self, **changes):
        params = {"from": "2026-11-13T10:00:00Z", "to": "2026-11-13T12:00:00Z", "bucket": "hour", "source": "field", **changes}
        reply = self.client.get("/api/v1/devices/uno-q-01/history", params=params)
        self.assertEqual(reply.status_code, 200, reply.text)
        return reply.json()

    def test_replay_is_acknowledged_without_incrementing_and_persists_after_restart(self):
        passage = self.event(assessment="mites_detected")
        self.assertEqual(self.upload(passage).json()["inserted"], 1)
        restart = TestClient(create_app(self.path, self.devices))
        self.addCleanup(restart.close)
        replay = self.upload(passage, client=restart).json()
        self.assertEqual(replay["committed_ids"], [passage["event_id"]])
        self.assertEqual((replay["inserted"], replay["duplicates"]), (0, 1))
        self.assertEqual(restart.get("/api/v1/devices/uno-q-01/status").json()["last_received_at"], replay["received_at"])
        result = self.history()["summary"]
        self.assertEqual(result["bees_observed"], 1)
        self.assertEqual(result["bees_with_detected_mites"], 1)
        self.assertEqual(result["availability"], "events_only")
        self.assertFalse(result["rate_available"])

    def test_conflicts_roll_back_whole_batch(self):
        existing = self.event()
        self.assertEqual(self.upload(existing).status_code, 200)
        new = self.event()
        altered = {**existing, "assessment": "mites_detected"}
        self.assertEqual(self.upload(new, altered).status_code, 409)
        # New row earlier in the rejected batch did not commit.
        self.assertEqual(self.upload(new).json()["inserted"], 1)
        reused_sequence = {**new, "event_id": str(uuid4())}
        self.assertEqual(self.upload(reused_sequence).status_code, 409)

    def test_auth_unknown_fields_naive_times_future_times_int_overflow(self):
        self.assertEqual(self.upload(self.event(), token="bad").status_code, 401)
        for change in [{"occurred_at": "2026-11-13T10:30:00"}, {"unknown": 1}, {"sequence": 2**63},
                       {"occurred_at": "2026-11-13 10:30:00Z"}, {"method": "unknown_v9"},
                       {"occurred_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}]:
            self.assertEqual(self.upload(self.event(**change)).status_code, 422, change)
        self.assertIsNone(self.history()["summary"]["bees_observed"])

    def test_coverage_union_measured_zero_missing_and_half_open_boundaries(self):
        cover = self.event("coverage")
        cover_overlap = self.event("coverage", start_at="2026-11-13T10:15:00Z")
        outside = self.event(occurred_at="2026-11-13T12:00:00Z")
        self.assertEqual(self.upload(cover, cover_overlap, outside).status_code, 200)
        result = self.history()
        self.assertEqual(result["summary"]["recorded_seconds"], 3600)
        self.assertEqual(result["summary"]["bees_observed"], 0)
        self.assertEqual(result["buckets"][0]["bees_observed"], 0)
        self.assertIsNone(result["buckets"][1]["bees_observed"])

    def test_out_of_order_upload_and_source_filtering(self):
        late = self.event(assessment="uncertain", occurred_at="2026-11-13T11:30:00Z")
        earlier = self.event(assessment="mites_detected")
        demo = self.event(source="screen_demo", assessment="mites_detected")
        self.upload(late)
        self.upload(earlier, demo)
        result = self.history()
        self.assertEqual(result["summary"]["bees_observed"], 2)
        self.assertEqual(result["summary"]["bees_with_detected_mites"], 1)
        self.assertEqual(result["summary"]["bees_assessment_uncertain"], 1)
        self.assertEqual([b["bees_observed"] for b in result["buckets"]], [1, 1])

    def test_partial_upload_coverage_never_makes_invalid_rate(self):
        inside = self.event()
        outside = self.event(occurred_at="2026-11-13T11:30:00Z")
        self.upload(self.event("coverage"), inside, outside)
        summary = self.history()["summary"]
        self.assertEqual(summary["bees_observed"], 2)
        self.assertEqual(summary["events_outside_coverage"], 1)
        self.assertFalse(summary["rate_available"])
        self.upload(self.event("coverage", start_at="2026-11-13T11:00:00Z", end_at="2026-11-13T12:00:00Z", occurred_at="2026-11-13T12:00:00Z"))
        self.assertTrue(self.history()["summary"]["rate_available"])

    def test_fractional_timestamp_lower_boundary_and_offset_normalization(self):
        self.upload(self.event(occurred_at="2026-11-13T12:00:00.500000+02:00"))
        self.assertEqual(self.history()["summary"]["bees_observed"], 1)

    def test_dst_calendar_days_and_repeated_hours(self):
        result = self.history(**{"from": "2026-10-24T21:00:00Z", "to": "2026-10-25T22:00:00Z", "tz": "Europe/Helsinki", "bucket": "day"})
        self.assertEqual(len(result["buckets"]), 1)
        self.assertEqual(result["buckets"][0]["requested_seconds"], 25 * 3600)
        result = self.history(**{"from": "2026-10-24T21:00:00Z", "to": "2026-10-25T22:00:00Z", "tz": "Europe/Helsinki"})
        self.assertEqual(len(result["buckets"]), 25)
        repeated = [b["label"] for b in result["buckets"] if "03:00" in b["label"]]
        self.assertEqual(len(set(repeated)), 2)
        result = self.history(**{"from": "2026-03-28T22:00:00Z", "to": "2026-03-29T21:00:00Z", "tz": "Europe/Helsinki", "bucket": "day"})
        self.assertEqual(result["buckets"][0]["requested_seconds"], 23 * 3600)

    def test_offline_upload_receipt_does_not_claim_current_recording(self):
        self.upload(self.event("heartbeat", pending_events=7))
        status = self.client.get("/api/v1/devices/uno-q-01/status").json()
        self.assertTrue(status["stale"])
        self.assertEqual(status["status"], "offline")
        self.assertEqual(status["pending_events"], 7)
        self.assertNotEqual(status["last_received_at"], status["last_occurred_at"])

    def test_csv_parity_validation_and_body_limit(self):
        self.upload(self.event("coverage"), self.event(assessment="mites_detected"))
        result = self.history()
        params = {"from": "2026-11-13T10:00:00Z", "to": "2026-11-13T12:00:00Z", "bucket": "hour"}
        reply = self.client.get("/api/v1/devices/uno-q-01/export.csv", params=params)
        self.assertEqual(reply.status_code, 200, reply.text)
        rows = list(csv.DictReader(io.StringIO(reply.text)))
        self.assertEqual(int(rows[0]["bees_observed"]), result["buckets"][0]["bees_observed"])
        self.assertEqual(rows[1]["bees_observed"], "")
        self.assertEqual(self.client.post("/api/v1/ingest/batch", content=b"x" * (2 * 1024 * 1024 + 1)).status_code, 413)
        self.assertEqual(self.client.get("/api/v1/devices/uno-q-01/history", params={**params, "tz": "Not/AZone"}).status_code, 422)
        self.assertEqual(self.client.get("/api/v1/devices/missing/status").status_code, 404)

    def test_five_minute_buckets_not_detected_counts_sources_and_live_stream(self):
        devices = {"uno-q-01": {"token": "a-valid-device-secret", "name": "Hive 01",
                                "timezone": "Europe/Helsinki", "live_url": "http://169.254.100.2:7000"}}
        client = TestClient(create_app(self.path, devices))
        self.addCleanup(client.close)
        cov = self.event("coverage", source="screen_demo", start_at="2026-11-13T10:00:00Z",
                         end_at="2026-11-13T10:10:00Z", occurred_at="2026-11-13T10:10:00Z")
        clean = self.event(source="screen_demo", assessment="not_detected", occurred_at="2026-11-13T10:02:00Z")
        uncertain = self.event(source="screen_demo", assessment="uncertain", occurred_at="2026-11-13T10:03:00Z")
        mite = self.event(source="screen_demo", assessment="mites_detected", occurred_at="2026-11-13T10:07:00Z")
        self.assertEqual(self.upload(cov, clean, uncertain, mite, client=client).status_code, 200)

        res = client.get("/api/v1/devices/uno-q-01/history", params={
            "from": "2026-11-13T10:00:00Z", "to": "2026-11-13T10:15:00Z", "bucket": "5m", "source": "screen_demo",
        }).json()
        self.assertEqual(len(res["buckets"]), 3)
        self.assertEqual([b["bees_observed"] for b in res["buckets"]], [2, 1, None])
        self.assertEqual(res["summary"]["bees_not_detected"], 1)
        self.assertEqual(res["summary"]["bees_assessment_uncertain"], 1)
        self.assertEqual(res["summary"]["bees_with_detected_mites"], 1)

        st = client.get("/api/v1/devices/uno-q-01/status").json()
        self.assertEqual(st["active_source"], "screen_demo")
        self.assertIn("screen_demo", st["sources"])
        self.assertEqual(st["live_url"], "http://169.254.100.2:7000")

        sse = client.get("/api/v1/stream", params={"once": "true"})
        self.assertEqual(sse.status_code, 200)
        self.assertIn("event: connected", sse.text)
        self.assertIn('"seq":1', sse.text)


if __name__ == "__main__":
    unittest.main()
