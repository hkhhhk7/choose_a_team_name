"""AprilTag detection and 3D pose estimation (CPU)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pupil_apriltags import Detector

from .config import load_yaml
from .events import TagPose


class Camera:
    """Pinhole intrinsics plus lens distortion."""

    def __init__(self, K: np.ndarray, dist: Optional[np.ndarray] = None, calibrated: bool = True):
        self.K = np.asarray(K, dtype=float)
        self.dist = None if dist is None else np.asarray(dist, dtype=float)
        self.calibrated = calibrated

    @property
    def params(self) -> tuple[float, float, float, float]:
        return self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

    @classmethod
    def load(cls, path: str | Path) -> "Camera":
        d = load_yaml(path)
        return cls(np.array(d["K"]), np.array(d.get("dist", [0, 0, 0, 0, 0])))

    @classmethod
    def guess(cls, width: int, height: int, hfov_deg: float = 65.0) -> "Camera":
        """Rough intrinsics for an uncalibrated webcam (typical ~65 deg FOV).

        Tilt angles stay usable; distances will be off by a few tens of
        percent until you run tools/calibrate_camera.py.
        """
        f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
        return cls(K, None, calibrated=False)


class TagDetector:
    """Detects tags and estimates their pose.

    pupil_apriltags takes one tag size per call. Translation scales linearly
    with the tag size (rotation does not), so tags of another size are fixed
    up afterwards: t_true = t_est * size_true / size_assumed.
    """

    def __init__(self, camera: Camera, tag_size_m: float, family: str = "tag36h11",
                 tag_sizes: Optional[dict[int, float]] = None,
                 decimate: float = 1.0, threads: int = 2):
        self.camera = camera
        self.tag_size_m = tag_size_m
        self.tag_sizes = tag_sizes or {}
        self.det = Detector(families=family, nthreads=threads, quad_decimate=decimate)
        self._maps = None

    def _undistort(self, gray: np.ndarray) -> np.ndarray:
        cam = self.camera
        if cam.dist is None or not np.any(cam.dist):
            return gray
        if self._maps is None or self._maps[2] != gray.shape:
            m1, m2 = cv2.initUndistortRectifyMap(cam.K, cam.dist, None, cam.K,
                                                 gray.shape[::-1], cv2.CV_16SC2)
            self._maps = (m1, m2, gray.shape)
        return cv2.remap(gray, self._maps[0], self._maps[1], cv2.INTER_LINEAR)

    def detect(self, frame_bgr: np.ndarray) -> dict[int, TagPose]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = self._undistort(gray)
        dets = self.det.detect(gray, estimate_tag_pose=True,
                               camera_params=self.camera.params, tag_size=self.tag_size_m)
        poses: dict[int, TagPose] = {}
        for d in dets:
            if d.hamming > 1 or d.decision_margin < 20:
                continue  # weak detection, likely a false positive
            tid = int(d.tag_id)
            scale = self.tag_sizes.get(tid, self.tag_size_m) / self.tag_size_m
            poses[tid] = TagPose(tid, np.asarray(d.pose_R),
                                 np.asarray(d.pose_t).reshape(3) * scale,
                                 np.asarray(d.corners))
        return poses
