"""Calibrate a webcam with a printed chessboard and write configs/camera.yaml.

    python tools/calibrate_camera.py --source 0 --cols 9 --rows 6 --square-mm 25

--cols/--rows count the INNER corners (a 10x7-square board has 9x6).
Hold the board at different angles and distances; press SPACE to capture a
view when the corners are drawn, c to calibrate (needs >= 12 views), q to quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default="configs/camera.yaml")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    pattern = (args.cols, args.rows)
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm / 1000.0
    obj_pts, img_pts = [], []
    size = None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"views: {len(obj_pts)}  SPACE capture, c calibrate, q quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_pts.append(objp)
            img_pts.append(corners)
        elif key == ord("c") and len(obj_pts) >= 12:
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()
    if len(obj_pts) < 12:
        raise SystemExit("need at least 12 views")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"RMS reprojection error: {rms:.3f} px (good: < 0.5)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"image_size": list(size), "rms_px": float(rms),
                        "K": K.tolist(), "dist": dist.ravel().tolist()}, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
