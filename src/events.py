"""Turn per-frame tag poses into bench events.

Work-area model: the camera watches only the area where the student works.
Bottles and beakers live outside the view and are brought in when needed, so
the events are

  ENTER / LEAVE   a bottle or a container comes into / goes out of view
  POUR            a bottle in view tips past pour_angle_deg near a container in view
  CAP_OPEN/CLOSE  a bottle's cap tag moves away from / back to where it sat

Geometry (metres):
  * AprilTag pose = rotation R (tag frame -> reference frame) and translation t.
  * Reference frame = the TABLE tag when one is configured and seen (it is
    mounted upright and never moves), else the camera.
  * Tags are stuck upright, so an object's "up" is R @ up_axis (up_axis = -y,
    the tag's y axis points down the printed tag). The reference's own up is
    up_axis too, so tilt = angle(R @ up_axis, up_axis).

Every rule needs several consecutive frames before it fires, leaving needs
leave_s without a sighting (a hand briefly covering a tag is not a "leave"),
and pouring uses hysteresis, so one noisy frame never triggers an event.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from .records import CAP_CLOSE, CAP_OPEN, ENTER, LEAVE, POUR, Event


@dataclass
class TagPose:
    tag_id: int
    R: np.ndarray  # 3x3, tag frame -> camera frame
    t: np.ndarray  # (3,), metres, camera frame
    corners: Optional[np.ndarray] = None  # 4x2 image points, for drawing


@dataclass
class EventParams:
    pour_angle_deg: float = 60.0
    pour_release_deg: float = 45.0
    pour_distance_cm: float = 15.0
    min_frames: int = 5
    enter_frames: int = 5
    leave_s: float = 1.5
    cap_open_distance_cm: float = 3.0
    cap_close_distance_cm: float = 1.5
    up_axis: tuple = (0.0, -1.0, 0.0)

    @classmethod
    def from_dict(cls, d: dict) -> "EventParams":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "up_axis" in kw:
            kw["up_axis"] = tuple(kw["up_axis"])
        return cls(**kw)


def tilt_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _as_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(i) for i in x]
    return [int(x)]


@dataclass
class Track:
    """One bottle or container."""

    name: str
    kind: str  # "bottle" | "container"
    tags: list
    cap_tag: Optional[int] = None
    present: bool = False
    seen_count: int = 0
    last_seen: float = -1e9
    pos: Optional[np.ndarray] = None
    # bottles only
    tilt: Optional[float] = None
    pour_count: int = 0
    release_count: int = 0
    pouring: bool = False
    nearest: Optional[str] = None
    nearest_cm: Optional[float] = None
    cap_base: Optional[float] = None
    cap_open: bool = False
    cap_count: int = 0


class EventExtractor:
    def __init__(self, lab: dict, params: Optional[EventParams] = None):
        self.p = params or EventParams()
        self.up_axis = np.asarray(self.p.up_axis, dtype=float)
        self.tracks: dict[str, Track] = {}
        for b in lab.get("bottles", []):
            tags = _as_list(b.get("body_tags", b.get("body_tag")))
            self.tracks[b["name"]] = Track(b["name"], "bottle", tags, b.get("cap_tag"))
        for c in lab.get("containers", []):
            self.tracks[c["name"]] = Track(c["name"], "container", _as_list(c["tag"]))
        # Optional reference tag fixed in the scene, mounted upright and facing
        # the camera. Poses are then expressed in its frame: a bumped camera is
        # not mistaken for motion, and "up" is the real vertical.
        self.table_tag: Optional[int] = lab.get("table_tag")
        self.table_pose: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._table_moved = 0

    # ------------------------------------------------------------- state

    @property
    def reference(self) -> str:
        if self.table_tag is None:
            return "camera (keep it level and still)"
        return "table tag" if self.table_pose is not None else "camera: table tag not seen yet"

    def in_view(self, kind: Optional[str] = None) -> list[str]:
        return [tr.name for tr in self.tracks.values()
                if tr.present and (kind is None or tr.kind == kind)]

    # -------------------------------------------------- reference frame

    def _update_table_pose(self, ref: TagPose) -> None:
        """Keep the reference pose fixed; re-anchor only when the camera really moved.

        Re-reading the reference every frame would add its jitter to every
        object. A change has to exceed 1 cm or 1.5 degrees for 3 frames in a
        row (fewer than min_frames, so a bump is absorbed before it can look
        like a pour).
        """
        R = np.asarray(ref.R, dtype=float)
        tv = np.asarray(ref.t, dtype=float)
        if self.table_pose is None:
            self.table_pose = (R, tv)
            return
        R0, t0 = self.table_pose
        moved = (np.linalg.norm(tv - t0) * 100 > 1.0
                 or tilt_deg(R0[:, 2], R[:, 2]) > 1.5 or tilt_deg(R0[:, 0], R[:, 0]) > 1.5)
        self._table_moved = self._table_moved + 1 if moved else 0
        if self._table_moved >= 3:
            self.table_pose = (R, tv)
            self._table_moved = 0

    def to_table_frame(self, poses: dict[int, TagPose]) -> dict[int, TagPose]:
        """Re-express poses relative to the table tag (camera frame until it is seen)."""
        if self.table_tag is None:
            return poses
        if self.table_tag in poses:
            first = self.table_pose is None
            self._update_table_pose(poses[self.table_tag])
            if first:  # positions stored in the camera frame are now meaningless
                for tr in self.tracks.values():
                    tr.pos = None
        if self.table_pose is None:
            return poses
        Rt, tt = self.table_pose
        out = {}
        for tid, pz in poses.items():
            if tid == self.table_tag:
                continue
            out[tid] = TagPose(tid, Rt.T @ np.asarray(pz.R, dtype=float),
                               Rt.T @ (np.asarray(pz.t, dtype=float) - tt), pz.corners)
        return out

    # ----------------------------------------------------------- update

    def update(self, t: float, poses: dict[int, TagPose]) -> list[Event]:
        poses = self.to_table_frame(poses)
        out: list[Event] = []
        # containers first, so a bottle pouring this frame sees them
        for tr in self.tracks.values():
            if tr.kind == "container":
                out += self._presence(t, tr, poses)
        for tr in self.tracks.values():
            if tr.kind == "bottle":
                out += self._presence(t, tr, poses)
                out += self._bottle(t, tr, poses)
        return out

    def _presence(self, t: float, tr: Track, poses: dict[int, TagPose]) -> list[Event]:
        visible = [poses[i] for i in tr.tags if i in poses]
        field = "bottle" if tr.kind == "bottle" else "container"
        if visible:
            tr.pos = np.asarray(visible[0].t, dtype=float)
            tr.last_seen = t
            tr.seen_count += 1
            if not tr.present and tr.seen_count >= self.p.enter_frames:
                tr.present = True
                return [Event(t, ENTER, **{field: tr.name})]
            return []
        if not tr.present:
            tr.seen_count = 0  # entering needs consecutive sightings
            return []
        if t - tr.last_seen >= self.p.leave_s:
            tr.present = False
            tr.seen_count = 0
            tr.pouring = False
            tr.pour_count = tr.release_count = 0
            tr.cap_base = None
            return [Event(t, LEAVE, **{field: tr.name})]
        return []  # briefly hidden (a hand over the tag): still in view

    def _bottle(self, t: float, tr: Track, poses: dict[int, TagPose]) -> list[Event]:
        visible = [poses[i] for i in tr.tags if i in poses]
        if not visible or not tr.present:
            return []
        p = self.p
        pose = visible[0]
        up = np.asarray(pose.R, dtype=float) @ self.up_axis
        tilt = tilt_deg(up, self.up_axis)
        tr.tilt = tilt

        nearest, dist_cm = self._nearest_container(tr.pos)
        tr.nearest, tr.nearest_cm = nearest, dist_cm

        out: list[Event] = []
        if not tr.pouring:
            near = nearest is not None and dist_cm <= p.pour_distance_cm
            tr.pour_count = tr.pour_count + 1 if (tilt >= p.pour_angle_deg and near) else 0
            if tr.pour_count >= p.min_frames:
                tr.pouring, tr.pour_count, tr.release_count = True, 0, 0
                out.append(Event(t, POUR, tr.name, nearest,
                                 meta={"tilt_deg": round(tilt, 1), "distance_cm": round(dist_cm, 1)}))
        else:
            # one pour lasts until the bottle is clearly upright again
            tr.release_count = tr.release_count + 1 if tilt < p.pour_release_deg else 0
            if tr.release_count >= p.min_frames:
                tr.pouring, tr.release_count = False, 0

        out += self._cap(t, tr, poses)
        return out

    def _nearest_container(self, pos: np.ndarray) -> tuple[Optional[str], Optional[float]]:
        best, best_d = None, None
        for tr in self.tracks.values():
            if tr.kind != "container" or not tr.present or tr.pos is None:
                continue
            d = float(np.linalg.norm(pos - tr.pos) * 100)
            if best_d is None or d < best_d:
                best, best_d = tr.name, d
        return best, best_d

    def _cap(self, t: float, tr: Track, poses: dict[int, TagPose]) -> list[Event]:
        """The cap is assumed closed when the bottle first shows both tags."""
        if tr.cap_tag is None or tr.cap_tag not in poses:
            return []
        d = float(np.linalg.norm(np.asarray(poses[tr.cap_tag].t) - tr.pos) * 100)
        if tr.cap_base is None:
            tr.cap_base = d
            return []
        off = abs(d - tr.cap_base)
        if not tr.cap_open:
            tr.cap_count = tr.cap_count + 1 if off > self.p.cap_open_distance_cm else 0
            if tr.cap_count >= self.p.min_frames:
                tr.cap_open, tr.cap_count = True, 0
                return [Event(t, CAP_OPEN, tr.name)]
        else:
            tr.cap_count = tr.cap_count + 1 if off < self.p.cap_close_distance_cm else 0
            if tr.cap_count >= self.p.min_frames:
                tr.cap_open, tr.cap_count = False, 0
                return [Event(t, CAP_CLOSE, tr.name)]
        return []
