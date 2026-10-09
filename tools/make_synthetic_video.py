"""Render a synthetic work-area video with moving AprilTags, for end-to-end tests.

    python tools/make_synthetic_video.py --scenario wrong_order --out samples/synthetic_wrong_order.mp4
    python -m src.main --source samples/synthetic_wrong_order.mp4 --no-display --record out.mp4

The camera looks straight at the work area. Bottles start outside the view,
are brought in, tipped sideways over a beaker (a rotation in the image plane,
which is how the real demo should be staged too) and taken away again.
The TABLE reference tag sits upright at the back and never moves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_lab  # noqa: E402

W, H, FPS, HFOV = 1280, 720, 30, 65.0
F = (W / 2) / np.tan(np.radians(HFOV) / 2)
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])

Z = 0.50  # work-area distance from the camera (m)
CAP_OFFSET = np.array([0.0, -0.07, 0.0])  # cap flag sits 7 cm above the body tag
OFF_LEFT, OFF_RIGHT = -0.45, 0.45  # outside the view
TABLE_AT = (-0.25, -0.15)
BEAKER_AT = {"beaker_A": (0.10, 0.09), "beaker_B": (0.26, 0.09)}
HOLD_AT = (-0.12, 0.0)  # where a bottle is held after being brought in


def pour_at(beaker: str) -> tuple[float, float]:
    bx, by = BEAKER_AT[beaker]
    return bx - 0.05, by - 0.06  # body tag about 8 cm from the beaker tag


def bottle_visit(start: float, beaker: str) -> list[tuple[float, float, float, float]]:
    """(time, x, y, tilt) keyframes: bring in, pour into `beaker`, take away."""
    hx, hy = HOLD_AT
    px, py = pour_at(beaker)
    return [
        (0.0, OFF_LEFT, hy, 0), (start, OFF_LEFT, hy, 0),
        (start + 1.0, hx, hy, 0),  # brought into view
        (start + 2.5, hx, hy, 0),  # read the screen
        (start + 3.5, px, py, 0),  # over the beaker
        (start + 4.3, px, py, 85),  # tip
        (start + 5.5, px, py, 85),  # pour
        (start + 6.0, px, py, 0),  # upright
        (start + 7.0, OFF_LEFT, hy, 0),  # taken away
        (99.0, OFF_LEFT, hy, 0),
    ]


def scenario(name: str) -> dict[str, list]:
    still = lambda x, y: [(0.0, x, y, 0), (99.0, x, y, 0)]  # noqa: E731
    k = {"beaker_A": still(*BEAKER_AT["beaker_A"]),
         "beaker_B": still(OFF_RIGHT, BEAKER_AT["beaker_B"][1])}
    if name == "correct":
        k["water"] = bottle_visit(1.0, "beaker_A")
        k["acid_conc"] = bottle_visit(9.0, "beaker_A")
    elif name == "wrong_order":
        k["acid_conc"] = bottle_visit(1.0, "beaker_A")
        k["water"] = bottle_visit(9.0, "beaker_A")
    elif name == "wrong_beaker":
        k["water"] = bottle_visit(1.0, "beaker_A")
        k["acid_conc"] = bottle_visit(9.0, "beaker_B")
        bx, by = BEAKER_AT["beaker_B"]
        k["beaker_B"] = [(0.0, OFF_RIGHT, by, 0), (9.5, OFF_RIGHT, by, 0), (10.5, bx, by, 0), (99.0, bx, by, 0)]
    return k


def interp(kf, t):
    ts = [k[0] for k in kf]
    i = max(0, min(len(kf) - 2, np.searchsorted(ts, t) - 1))
    (t0, *a), (t1, *b) = kf[i], kf[i + 1]
    s = 0.0 if t1 == t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    s = s * s * (3 - 2 * s)  # ease in/out
    return [x + (y - x) * s for x, y in zip(a, b)]


def rot_z(deg: float) -> np.ndarray:
    """Rotation in the image plane; positive tips the bottle top to the right."""
    a = np.radians(-deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def tag_image(dictionary, tag_id: int) -> np.ndarray:
    img = cv2.aruco.generateImageMarker(dictionary, tag_id, 80)  # 8 cells x 10 px
    return cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def draw_tag(frame, img, size_m, R, t):
    full = size_m * 10 / 8  # tag plus white border
    n = img.shape[0]
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    obj = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * full / 2  # x right, y down
    cam = (R @ obj.T).T + t
    uv = (K @ cam.T).T
    dst = np.float32(uv[:, :2] / uv[:, 2:])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(img, 255), Hm, (W, H), flags=cv2.INTER_NEAREST)
    frame[mask > 0] = warped[mask > 0, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["correct", "wrong_order", "wrong_beaker"], default="wrong_order")
    ap.add_argument("--lab", default="configs/bottles.yaml")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--out", default="samples/synthetic_wrong_order.mp4")
    ap.add_argument("--bump-at", type=float, help="the camera gets knocked (3 deg, 2 cm) at this time")
    args = ap.parse_args()

    lab = load_lab(args.lab)
    bottles = {b["name"]: b for b in lab["bottles"]}
    containers = {c["name"]: c for c in lab["containers"]}
    sizes = lab["tag_sizes"]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    imgs = {tid: tag_image(dictionary, tid) for tid in sizes}
    kf = scenario(args.scenario)
    rng = np.random.default_rng(0)

    a = np.radians(3)
    G_bump = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    s_bump = np.array([0.02, 0.005, 0.0])

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(int(args.seconds * FPS)):
        t = i / FPS
        frame = np.full((H, W, 3), (170, 160, 150), np.uint8)  # bench colour
        bumped = args.bump_at is not None and t >= args.bump_at

        def put(tag_id, R, pos):
            if bumped:  # everything moves together when the camera moves
                R, pos = G_bump @ R, G_bump @ pos + s_bump
            draw_tag(frame, imgs[tag_id], sizes[tag_id], R, pos)

        if lab.get("table_tag") is not None:
            put(lab["table_tag"], np.eye(3), np.array([*TABLE_AT, Z]))
        for name, c in containers.items():
            x, y, _ = interp(kf[name], t)
            put(c["tag"], np.eye(3), np.array([x, y, Z]))
        for name in ("water", "acid_conc"):
            if name not in kf:
                continue
            x, y, tilt = interp(kf[name], t)
            R = rot_z(tilt)
            body = np.array([x, y, Z])
            b = bottles[name]
            put(b["body_tag"], R, body)
            put(b["cap_tag"], R, body + R @ CAP_OFFSET)
        noise = rng.normal(0, 4, frame.shape)
        frame = np.clip(cv2.GaussianBlur(frame, (3, 3), 0) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    print(f"wrote {args.out} ({args.seconds:g} s, {args.scenario})")


if __name__ == "__main__":
    main()
