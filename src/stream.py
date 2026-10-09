"""Live video sources that never lag behind the bench and survive Wi-Fi hiccups.

LatestFrame  wraps any cv2.VideoCapture (USB webcam, rtsp://) on a thread.
MjpegStream  reads an http:// MJPEG stream (phone camera apps such as
             Android "IP Webcam") itself instead of through FFmpeg: a glitch
             in the stream made FFmpeg abort the whole process, while here it
             only triggers a reconnect.
Both keep only the newest frame, so a slow consumer never sees stale video.
"""
from __future__ import annotations

import threading
import time
import urllib.request

import cv2
import numpy as np


class _Latest:
    def __init__(self):
        self.frame = None
        self.seq = 0
        self.running = True
        self.cond = threading.Condition()

    def _publish(self, frame) -> None:
        with self.cond:
            self.frame = frame
            self.seq += 1
            self.cond.notify_all()

    def read(self, timeout: float = 2.0):
        """Newest frame; if none arrives within `timeout`, repeat the last one."""
        with self.cond:
            seen = self.seq
            self.cond.wait_for(lambda: self.seq != seen or not self.running, timeout)
            if self.frame is None:
                return False, None
            return True, self.frame

    def get(self, prop):
        return 0.0

    def release(self) -> None:
        self.running = False
        with self.cond:
            self.cond.notify_all()


class LatestFrame(_Latest):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__()
        self.cap = cap
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            self._publish(frame)

    def get(self, prop):
        return self.cap.get(prop)

    def release(self) -> None:
        super().release()
        self.cap.release()


class MjpegStream(_Latest):
    def __init__(self, url: str, connect_timeout: float = 5.0):
        super().__init__()
        self.url = url
        self.connect_timeout = connect_timeout
        self.reconnects = 0
        threading.Thread(target=self._run, daemon=True).start()

    def wait_first(self, timeout: float = 10.0) -> bool:
        with self.cond:
            return self.cond.wait_for(lambda: self.frame is not None, timeout)

    def _run(self) -> None:
        while self.running:
            try:
                with urllib.request.urlopen(self.url, timeout=self.connect_timeout) as resp:
                    buf = bytearray()
                    while self.running:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        buf += chunk
                        frame = None
                        # keep only the newest complete JPEG in the buffer
                        while True:
                            start = buf.find(b"\xff\xd8")
                            end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                            if start < 0 or end < 0:
                                if start > 0:
                                    del buf[:start]
                                break
                            jpg = bytes(buf[start:end + 2])
                            del buf[:end + 2]
                            frame = jpg
                        if frame is not None:
                            img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                self._publish(img)
                        if len(buf) > 8 * 1024 * 1024:  # garbage, start over
                            buf.clear()
            except Exception as e:  # network error: reconnect
                if self.running and self.reconnects % 20 == 0:
                    print(f"[stream] {type(e).__name__}: {e}; reconnecting...")
            if self.running:
                self.reconnects += 1
                time.sleep(0.5)
