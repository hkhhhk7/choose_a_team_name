"""Plain data records shared by every module (no third-party imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Event kinds produced by the event extractor (src/events.py).
ENTER = "enter"  # a bottle or container comes into the work area
LEAVE = "leave"  # ... and goes out of it
POUR = "pour"
CAP_OPEN = "cap_open"
CAP_CLOSE = "cap_close"
EVENT_KINDS = (ENTER, LEAVE, POUR, CAP_OPEN, CAP_CLOSE)

# Alert levels, from least to most severe.
INFO = "info"
WARNING = "warning"
DANGER = "danger"
LEVELS = (INFO, WARNING, DANGER)


@dataclass
class Event:
    """Something a student did at the bench, derived from tag motion."""

    t: float  # seconds since the session started
    kind: str  # one of EVENT_KINDS
    bottle: Optional[str] = None
    container: Optional[str] = None  # POUR target, or the container that entered/left
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind: {self.kind!r}")


@dataclass
class Alert:
    """A message the SOP checker wants to show the student."""

    t: float
    level: str  # one of LEVELS
    code: str  # machine-readable reason, e.g. "OUT_OF_ORDER"
    message: str
    bottle: Optional[str] = None
    container: Optional[str] = None
    step: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.t:7.2f}s] {self.level.upper():7s} {self.code}: {self.message}"
