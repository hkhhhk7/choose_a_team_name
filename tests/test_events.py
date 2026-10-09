"""Event extraction on synthetic tag poses (no camera needed)."""
import unittest

import numpy as np

from src.events import EventExtractor, EventParams, TagPose
from src.records import CAP_CLOSE, CAP_OPEN, ENTER, LEAVE, POUR

LAB = {
    "bottles": [{"name": "water", "body_tag": 1, "cap_tag": 2}],
    "containers": [{"name": "beaker_A", "tag": 10}, {"name": "beaker_B", "tag": 11}],
}
LAB_T = dict(LAB, table_tag=20)

PARAMS = EventParams(min_frames=3, enter_frames=3, leave_s=0.5, pour_distance_cm=12)
FPS = 30


def rot_z(deg: float) -> np.ndarray:
    """Tag turning in the image plane: how a bottle facing the camera tips sideways."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class Scene:
    """Camera frame, metres; the beakers sit 30 cm in front of the camera."""

    BEAKERS = {10: np.array([0.15, 0.0, 0.30]), 11: np.array([-0.15, 0.0, 0.30])}

    def __init__(self, lab=LAB):
        self.ex = EventExtractor(lab, PARAMS)
        self.t = 0.0
        self.events = []

    def frame(self, bottle=None, tilt=0.0, beakers=(10,), cap=(0.0, -0.06, 0.0), n=1):
        for _ in range(n):
            self.t += 1 / FPS
            poses = {b: TagPose(b, np.eye(3), self.BEAKERS[b]) for b in beakers}
            if bottle is not None:
                R = rot_z(tilt)
                poses[1] = TagPose(1, R, np.array(bottle, float))
                if cap is not None:
                    poses[2] = TagPose(2, R, np.array(bottle, float) + R @ np.array(cap))
            self.events += self.ex.update(self.t, poses)

    def kinds(self):
        return [(e.kind, e.bottle or e.container) for e in self.events]


class EventExtractorTest(unittest.TestCase):
    def test_enter_needs_consecutive_frames(self):
        s = Scene()
        s.frame(n=2)
        self.assertEqual(s.events, [])
        s.frame(n=1)
        self.assertEqual(s.kinds(), [(ENTER, "beaker_A")])

    def test_bottle_enters_pours_and_leaves(self):
        s = Scene()
        s.frame(n=3)  # beaker in view
        s.frame(bottle=[0.05, -0.05, 0.3], n=3)  # bottle brought in
        self.assertEqual(s.kinds(), [(ENTER, "beaker_A"), (ENTER, "water")])
        s.frame(bottle=[0.10, -0.05, 0.3], tilt=75, n=3)  # tip over the beaker
        self.assertEqual(s.events[-1].kind, POUR)
        self.assertEqual(s.events[-1].container, "beaker_A")
        s.frame(bottle=[0.10, -0.05, 0.3], tilt=85, n=20)  # keep pouring: still one pour
        s.frame(bottle=[0.10, -0.05, 0.3], tilt=50, n=5)  # wobble above release: same pour
        s.frame(bottle=[0.10, -0.05, 0.3], tilt=80, n=5)
        self.assertEqual([k for k, _ in s.kinds()].count(POUR), 1)
        s.frame(n=20)  # bottle taken away
        self.assertEqual(s.kinds()[-1], (LEAVE, "water"))

    def test_pour_goes_to_nearest_beaker_in_view(self):
        s = Scene()
        s.frame(beakers=(10, 11), n=3)
        s.frame(bottle=[-0.10, -0.05, 0.3], beakers=(10, 11), n=3)
        s.frame(bottle=[-0.10, -0.05, 0.3], tilt=-75, beakers=(10, 11), n=3)
        self.assertEqual(s.events[-1].kind, POUR)
        self.assertEqual(s.events[-1].container, "beaker_B")

    def test_no_pour_without_a_beaker_nearby(self):
        s = Scene()
        s.frame(beakers=(), bottle=[0.0, 0.0, 0.3], tilt=80, n=10)
        self.assertNotIn(POUR, [k for k, _ in s.kinds()])

    def test_one_noisy_frame_does_not_trigger(self):
        s = Scene()
        s.frame(n=3)
        s.frame(bottle=[0.10, -0.05, 0.3], n=3)
        s.frame(bottle=[0.10, -0.05, 0.3], tilt=75, n=1)
        s.frame(bottle=[0.10, -0.05, 0.3], n=3)
        self.assertNotIn(POUR, [k for k, _ in s.kinds()])

    def test_brief_occlusion_is_not_a_leave(self):
        s = Scene()
        s.frame(n=3)
        s.frame(bottle=[0.05, -0.05, 0.3], n=3)
        s.frame(n=int(0.3 * FPS))  # hand over the tag for 0.3 s (< leave_s)
        s.frame(bottle=[0.05, -0.05, 0.3], n=3)
        self.assertEqual(s.kinds(), [(ENTER, "beaker_A"), (ENTER, "water")])

    def test_cap_open_and_close(self):
        s = Scene()
        s.frame(n=3)
        s.frame(bottle=[0.0, 0.0, 0.3], n=4)  # cap on: baseline
        s.frame(bottle=[0.0, 0.0, 0.3], cap=(0.15, 0.0, 0.0), n=4)  # cap set aside
        s.frame(bottle=[0.0, 0.0, 0.3], n=4)  # cap back on
        kinds = [k for k, _ in s.kinds()]
        self.assertEqual(kinds[-2:], [CAP_OPEN, CAP_CLOSE])


class TableReferenceTest(unittest.TestCase):
    """A knocked camera turns every tag at once; the table tag cancels that."""

    SHARP = EventParams(min_frames=3, enter_frames=3, leave_s=0.5, pour_distance_cm=12,
                        pour_angle_deg=25, pour_release_deg=15)

    def run_scene(self, lab, n=40, bump_at=15):
        ex = EventExtractor(lab, self.SHARP)
        G = rot_z(30)  # camera rolls 30 degrees about its view axis
        kinds = []
        for i in range(n):
            poses = {
                1: TagPose(1, np.eye(3), np.array([0.10, -0.05, 0.3])),  # bottle next to beaker
                10: TagPose(10, np.eye(3), np.array([0.15, 0.0, 0.3])),
                20: TagPose(20, np.eye(3), np.array([-0.10, 0.02, 0.35])),
            }
            if i >= bump_at:
                poses = {k: TagPose(k, G @ v.R, G @ v.t) for k, v in poses.items()}
            kinds += [e.kind for e in ex.update(i / FPS, poses)]
        return kinds

    def test_without_table_tag_a_rolled_camera_reads_as_tilt(self):
        self.assertIn(POUR, self.run_scene(LAB))

    def test_with_table_tag_a_bump_is_ignored(self):
        self.assertEqual(self.run_scene(LAB_T), [ENTER, ENTER])  # beaker, bottle; no pour

    def test_table_frame_still_detects_real_pour(self):
        ex = EventExtractor(LAB_T, PARAMS)
        kinds = []
        for i in range(20):
            tilt = 0 if i < 8 else 80
            poses = {1: TagPose(1, rot_z(tilt), np.array([0.10, -0.05, 0.3])),
                     10: TagPose(10, np.eye(3), np.array([0.15, 0.0, 0.3])),
                     20: TagPose(20, np.eye(3), np.array([-0.10, 0.02, 0.35]))}
            kinds += [e.kind for e in ex.update(i / FPS, poses)]
        self.assertEqual(kinds, [ENTER, ENTER, POUR])


if __name__ == "__main__":
    unittest.main()
