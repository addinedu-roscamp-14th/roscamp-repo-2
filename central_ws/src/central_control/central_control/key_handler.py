from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KeyAction(str, Enum):
    STEP = "STEP"
    HOME = "HOME"
    STOP = "STOP"
    RESET = "RESET"
    BYPASS_PARKING = "BYPASS_PARKING"
    QUIT = "QUIT"
    IGNORED = "IGNORED"


@dataclass(frozen=True)
class KeyDecision:
    action: KeyAction
    reason: str = ""


class KeyHandler:
    def decide(self, key: int, busy: bool = False) -> KeyDecision | None:
        if key in (-1, 255):
            return None
        if key in (ord("s"), ord("S")):
            return KeyDecision(KeyAction.STOP, "STOP has priority")
        if key in (ord("q"), ord("Q"), 27):
            return KeyDecision(KeyAction.QUIT, "quit requested")
        if busy and key in (32, 13, 10):
            return KeyDecision(KeyAction.IGNORED, "busy: SPACE ignored")
        if key in (32, 13, 10):
            return KeyDecision(KeyAction.STEP, "manual step")
        if key in (ord("h"), ord("H")):
            return KeyDecision(KeyAction.HOME, "manual HOME")
        if key in (ord("r"), ord("R")):
            return KeyDecision(KeyAction.RESET, "reset")
        if key in (ord("n"), ord("N")):
            return KeyDecision(KeyAction.BYPASS_PARKING, "bypass parking")
        return KeyDecision(KeyAction.IGNORED, f"ignored key: {key}")
