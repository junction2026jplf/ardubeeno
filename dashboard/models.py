"""Version-one wire events: timestamps are capture time, never upload time."""
from datetime import datetime, timedelta, timezone
import re
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UTC = timezone.utc
Source = Literal["field", "screen_demo", "replay"]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event_id: UUID
    session_id: UUID
    sequence: int = Field(ge=1, le=9223372036854775807)
    occurred_at: datetime
    source: Source
    method: Literal["bee_tracks_v1"] = "bee_tracks_v1"

    @field_validator("event_id", "session_id", mode="before")
    @classmethod
    def uuid_from_json(cls, value):
        return UUID(value) if isinstance(value, str) else value

    @field_validator("occurred_at", mode="before")
    @classmethod
    def parse_time(cls, value):
        return parse_timestamp(value)


def parse_timestamp(value):
    if isinstance(value, str):
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})", value):
                raise ValueError
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Expected an RFC3339 timestamp with a UTC offset") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware timestamp is required")
    value = value.astimezone(UTC)
    if value > datetime.now(UTC) + timedelta(minutes=5):
        raise ValueError("Timestamp is more than five minutes in the future")
    return value


class BeePassage(Event):
    kind: Literal["bee_passage"]
    assessment: Literal["mites_detected", "not_detected", "uncertain"]


class Coverage(Event):
    kind: Literal["coverage"]
    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at", mode="before")
    @classmethod
    def interval_time(cls, value):
        return parse_timestamp(value)

    @model_validator(mode="after")
    def check_interval(self):
        if not self.start_at < self.end_at:
            raise ValueError("Coverage start_at must precede end_at")
        if self.end_at - self.start_at > timedelta(days=366):
            raise ValueError("A coverage interval cannot exceed 366 days")
        if self.end_at != self.occurred_at:
            raise ValueError("Coverage occurred_at must equal end_at")
        return self


class Heartbeat(Event):
    kind: Literal["heartbeat"]
    status: Literal["recording", "idle", "camera_error", "storage_error"]
    pending_events: int = Field(default=0, ge=0, le=9223372036854775807)


WireEvent = Annotated[BeePassage | Coverage | Heartbeat, Field(discriminator="kind")]


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1]
    device_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    events: list[WireEvent] = Field(min_length=1, max_length=500)


def stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
