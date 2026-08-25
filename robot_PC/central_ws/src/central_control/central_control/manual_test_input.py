"""Non-blocking terminal input and queued fake vision results for test mode."""
from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
import uuid

from central_control.vision_client import VisionClientResult


MANUAL_TEST_HELP = """=== MANUAL TEST MODE ===
[p] vehicle parked/stable
[1] GOOD / GOOD
[2] BAD / GOOD
[3] GOOD / BAD
[4] BAD / BAD
[t] transport ready
[h] status
[s] stop
[e] emergency stop
[r] reset
[q] quit keyboard input"""


def manual_detection_response(left_status: str, right_status: str) -> dict:
    def side_result(status):
        return {
            "status": status,
            "class_name": "good_tire" if status == "GOOD" else "bad_tire",
            "confidence": 1.0,
            "stable": True,
            "votes": 1,
            "frames": 1,
            "detections": [],
        }

    return {
        "status": "ok",
        "type": "detection_result",
        "left": side_result(left_status),
        "right": side_result(right_status),
    }


class ManualTestVisionClient:
    """A single-result perception adapter consumed by the normal vision worker."""

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = None
        self._closed = False
        self._generation = 0
        self.calls = 0

    def submit_detection(self, left_status: str, right_status: str) -> bool:
        response = manual_detection_response(left_status, right_status)
        with self._condition:
            if self._closed or self._pending is not None:
                return False
            self._pending = response
            self._condition.notify_all()
        return True

    def request_detection(self, reset_cycle: bool = True) -> VisionClientResult:
        del reset_cycle
        request_id = f"manual-detect-{uuid.uuid4()}"
        with self._condition:
            generation = self._generation
            self.calls += 1
            while (
                self._pending is None
                and not self._closed
                and generation == self._generation
            ):
                self._condition.wait(timeout=0.1)
            if self._closed:
                return VisionClientResult(
                    False, error="manual test vision client closed", request_id=request_id
                )
            if generation != self._generation:
                return VisionClientResult(
                    False, error="manual test detection cancelled", request_id=request_id
                )
            response = self._pending
            self._pending = None
        return VisionClientResult(True, response=response, request_id=request_id)

    def clear(self):
        with self._condition:
            self._pending = None
            self._generation += 1
            self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class ManualTestKeyboard:
    """Read single terminal keys on a daemon thread without blocking ROS spin."""

    def __init__(self, on_key, stream=None):
        self.on_key = on_key
        self.stream = stream or sys.stdin
        self.stop_event = threading.Event()
        self.thread = None

    def start(self) -> bool:
        print(MANUAL_TEST_HELP, flush=True)
        try:
            fd = self.stream.fileno()
        except (AttributeError, OSError):
            print("[TEST] keyboard disabled: stdin has no file descriptor", flush=True)
            return False
        if not os.isatty(fd):
            print("[TEST] keyboard disabled: stdin is not a TTY", flush=True)
            return False
        self.thread = threading.Thread(
            target=self._run, args=(fd,), daemon=True, name="manual-test-keyboard"
        )
        self.thread.start()
        return True

    def _run(self, fd):
        original = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    continue
                key = os.read(fd, 1).decode(errors="ignore")
                if not key:
                    continue
                self.on_key(key)
                if key.lower() == "q":
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original)
            self.stop_event.set()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)

