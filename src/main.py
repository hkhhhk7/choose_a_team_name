"""LabGuardian: watch a bench through a camera and check it against an SOP.

    python -m src.main --source 0 --sop configs/sop/dilute_acid.yaml     # USB webcam
    python -m src.main --source http://192.168.1.23:8080/video --beep   # phone camera app over Wi-Fi
    python -m src.main --source samples/demo.mp4 --record out.mp4       # recorded video

Keys: q quit, r reset the procedure, s save the report now.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .config import load_lab, load_yaml
from .events import EventExtractor, EventParams
from .records import DANGER, INFO, WARNING, Alert
from .sop_checker import SOPChecker
from .stream import LatestFrame, MjpegStream
from .tags import Camera, TagDetector

COLORS = {INFO: (80, 200, 80), WARNING: (0, 165, 255), DANGER: (40, 40, 230)}  # BGR
ALERT_SHOW_S = 4.0  # how long an alert banner stays on screen
REPEAT_COOLDOWN_S = 5.0  # don't repeat the same alert sooner than this


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0",
                    help="camera index (0, 1, ...), phone stream URL (http://.../video) or video file")
    ap.add_argument("--sop", default="configs/sop/dilute_acid.yaml")
    ap.add_argument("--lab", default="configs/bottles.yaml")
    ap.add_argument("--camera", default="configs/camera.yaml",
                    help="intrinsics from tools/calibrate_camera.py (guessed if missing)")
    ap.add_argument("--hfov", type=float, default=65.0,
                    help="horizontal field of view used when there is no calibration file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--record", help='save the annotated video to this file ("auto": recordings/live_<time>.mp4)')
    ap.add_argument("--log", default="samples/session", help="prefix for events.jsonl and report.json")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-seconds", type=float, help="stop after this long (tests, unattended runs)")
    ap.add_argument("--beep", action="store_true", help="beep on danger alerts (Windows)")
    return ap.parse_args()


def open_source(src: str, width: int, height: int):
    """Camera index ("0"), phone camera URL (http://...), rtsp:// stream, or a video file.

    Returns (reader, live). Live readers keep only the newest frame.
    """
    low = src.lower()
    if src.isdigit():
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera {src}")
        return LatestFrame(cap), True
    if low.startswith(("http://", "https://")):
        if urlparse(src).path in ("", "/"):
            src = src.rstrip("/") + "/video"  # IP Webcam's MJPEG endpoint
        stream = MjpegStream(src)
        if not stream.wait_first(10.0):
            stream.release()
            raise SystemExit(f"no frames from {src}: is the phone app running on the same network?")
        return stream, True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {src!r}")
    if low.startswith("rtsp://"):
        return LatestFrame(cap), True
    return cap, False


def tag_owner(lab: dict, tag_id: int) -> str:
    if tag_id in lab["tag_to_bottle_body"]:
        return lab["tag_to_bottle_body"][tag_id]
    if tag_id in lab["tag_to_bottle_cap"]:
        return lab["tag_to_bottle_cap"][tag_id] + " cap"
    if tag_id == lab.get("table_tag"):
        return "table"
    return lab["tag_to_container"].get(tag_id, f"tag {tag_id}")


PANEL_W = 360
FONT = cv2.FONT_HERSHEY_SIMPLEX


def wrap(text: str, width_px: int, scale: float, thick: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > width_px and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + ([cur] if cur else [])


def draw(frame, poses, lab, extractor: EventExtractor, checker: SOPChecker,
         banners: list[Alert], now: float, fps: float):
    """Annotate the frame and return it with a status panel on the right."""
    h, w = frame.shape[:2]
    for tid, pose in poses.items():
        if pose.corners is None:
            continue
        pts = pose.corners.astype(int).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
        x, y = pose.corners[:, 0].min(), pose.corners[:, 1].min()
        label = tag_owner(lab, tid)
        tr = extractor.tracks.get(label)
        if tr is not None and tr.tilt is not None:
            label += f"  {tr.tilt:4.0f} deg"
            if tr.nearest_cm is not None:
                label += f"  {tr.nearest_cm:4.1f} cm"
        cv2.putText(frame, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 200, 0), 2, cv2.LINE_AA)

    # alert banners on the video, newest at the top
    by = 10
    for a in banners:
        if now - a.t > ALERT_SHOW_S:
            continue
        lines = wrap(a.message, w - 60, 0.75, 2)
        bh = 18 + 32 * len(lines)
        cv2.rectangle(frame, (10, by), (w - 10, by + bh), COLORS[a.level], -1)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (26, by + 36 + 32 * i), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        by += bh + 8

    # status panel to the right of the video
    panel = np.full((h, PANEL_W, 3), (32, 32, 32), np.uint8)
    x0, y = 16, 40
    cv2.putText(panel, "LabGuardian", (x0, y), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    y += 34
    for line in wrap(checker.name, PANEL_W - 2 * x0, 0.6, 1):
        cv2.putText(panel, line, (x0, y), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        y += 26
    y += 10
    nxt = {s.id for s in checker.next_steps()}
    for n, s in enumerate(checker.steps, 1):
        if s.id in checker.done:
            mark, col = "done", (80, 200, 80)
        elif s.id in nxt:
            mark, col = "NEXT", (0, 220, 255)
        else:
            mark, col = "wait", (150, 150, 150)
        text = f"{n}. {checker._label(s.reagent)} -> {checker._label(s.into)}"
        cv2.putText(panel, mark, (x0, y), FONT, 0.55, col, 2, cv2.LINE_AA)
        for i, line in enumerate(wrap(text, PANEL_W - x0 - 80, 0.55, 1)):
            cv2.putText(panel, line, (x0 + 62, y + 24 * i), FONT, 0.55, col, 1, cv2.LINE_AA)
            last = i
        y += 24 * (last + 1) + 12

    y += 10
    n_warn = sum(a.level == WARNING for a in checker.problems)
    n_err = sum(a.level == DANGER for a in checker.problems)
    cv2.putText(panel, f"warnings: {n_warn}   errors: {n_err}", (x0, y), FONT, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)
    y += 40
    cv2.putText(panel, "In the work area:", (x0, y), FONT, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    in_view = extractor.in_view()
    for line in wrap(", ".join(checker._label(n) for n in in_view) or "(nothing yet)",
                     PANEL_W - 2 * x0, 0.55, 1):
        y += 26
        cv2.putText(panel, line, (x0, y), FONT, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    cv2.putText(panel, f"reference: {extractor.reference}", (x0, h - 64), FONT, 0.45, (190, 190, 190), 1,
                cv2.LINE_AA)
    cv2.putText(panel, f"{fps:4.1f} fps on CPU", (x0, h - 40), FONT, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(panel, "processed on-device, frames not stored", (x0, h - 16), FONT, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)
    return np.hstack([frame, panel])


def main() -> None:
    args = parse_args()
    sop = load_yaml(args.sop)
    lab = load_lab(args.lab)

    cap, live = open_source(args.source, args.width, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("no frames from source")
    h, w = frame.shape[:2]

    if Path(args.camera).exists():
        camera = Camera.load(args.camera)
    else:
        camera = Camera.guess(w, h, args.hfov)
        print(f"[warn] {args.camera} not found; using guessed intrinsics. "
              "Run tools/calibrate_camera.py for accurate distances.")

    detector = TagDetector(camera, lab.get("tag_size_m", 0.030), lab.get("tag_family", "tag36h11"),
                           tag_sizes=lab["tag_sizes"])
    extractor = EventExtractor(lab, EventParams.from_dict(sop))
    checker = SOPChecker(sop, lab)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    written = 0
    if args.record:
        if args.record == "auto":  # recordings/live_YYYYmmdd_HHMMSS.mp4
            Path("recordings").mkdir(exist_ok=True)
            args.record = time.strftime("recordings/live_%Y%m%d_%H%M%S.mp4")
            if args.log == "samples/session":  # keep each run's events next to its video
                args.log = args.record[:-4]
        rec_fps = 30.0 if live else src_fps
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), rec_fps, (w + PANEL_W, h))
        print(f"[record] {args.record}")

    log_prefix = Path(args.log)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    event_log = open(f"{log_prefix}.events.jsonl", "w", encoding="utf-8")

    window_ready = False
    banners: list[Alert] = []
    last_shown: dict[tuple, float] = {}
    t0 = time.perf_counter()
    frame_idx = 0
    fps, last_tick = 0.0, time.perf_counter()

    def save_report() -> None:
        with open(f"{log_prefix}.report.json", "w", encoding="utf-8") as f:
            json.dump(checker.summary(), f, indent=2, ensure_ascii=False)

    try:
        while ok:
            # live: wall-clock time; video: frame time, so replays are deterministic
            now = time.perf_counter() - t0 if live else frame_idx / src_fps
            poses = detector.detect(frame)
            alerts: list[Alert] = []
            for ev in extractor.update(now, poses):
                event_log.write(json.dumps({"t": round(ev.t, 3), "kind": ev.kind, "bottle": ev.bottle,
                                            "container": ev.container, **ev.meta}) + "\n")
                alerts += checker.process(ev)
            tick_alerts = checker.tick(now)
            checker.alerts += tick_alerts
            alerts += tick_alerts

            for a in alerts:
                key = (a.code, a.bottle, a.container)
                if now - last_shown.get(key, -1e9) < REPEAT_COOLDOWN_S:
                    continue
                last_shown[key] = now
                print(a)
                banners.insert(0, a)
                if a.level == DANGER and args.beep:
                    try:
                        import winsound
                        winsound.Beep(1500, 300)
                    except Exception:
                        pass
            banners = banners[:3]

            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t - last_tick, 1e-6))
            last_tick = t

            view = draw(frame, poses, lab, extractor, checker, banners, now, fps)
            if writer is not None:
                if live:
                    # processing may run slower than 30 fps: repeat frames so the
                    # recording plays back at real speed
                    while written < int(now * rec_fps) + 1:
                        writer.write(view)
                        written += 1
                else:
                    writer.write(view)
            if not args.no_display:
                if not window_ready:  # resizable window that fits a laptop screen
                    cv2.namedWindow("LabGuardian", cv2.WINDOW_NORMAL)
                    scale = min(1.0, 1500 / view.shape[1])
                    cv2.resizeWindow("LabGuardian", int(view.shape[1] * scale), int(view.shape[0] * scale))
                    window_ready = True
                cv2.imshow("LabGuardian", view)
                key = cv2.waitKey(1) & 0xFF
                # q, Esc or the window's close button: stop cleanly so the recording is saved
                if key in (ord("q"), 27) or cv2.getWindowProperty("LabGuardian", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    checker.reset()
                    banners.clear()
                    print("[reset]")
                if key == ord("s"):
                    save_report()
                    print(f"[saved] {log_prefix}.report.json")
            if args.max_seconds is not None and now >= args.max_seconds:
                break
            ok, frame = cap.read()
            frame_idx += 1
    finally:
        save_report()
        event_log.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(json.dumps(checker.summary(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
