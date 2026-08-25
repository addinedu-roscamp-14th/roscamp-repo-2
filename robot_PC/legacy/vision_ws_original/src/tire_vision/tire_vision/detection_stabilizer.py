from dataclasses import dataclass
from collections import deque


@dataclass(frozen=True)
class StableResult:
    label: str
    confidence: float
    stable_count: int
    stable: bool


class DetectionStabilizer:
    def __init__(
        self,
        required_count: int = 3,
        bad_class_name: str = "bad_tire",
        good_class_name: str = "good_tire",
        window_size: int | None = None,
        minimum_average_confidence: float = 0.0,
    ):
        self.required_count = max(1, int(required_count))
        self.bad_class_name = bad_class_name
        self.good_class_name = good_class_name
        self.window_size = max(self.required_count, int(window_size or self.required_count))
        self.minimum_average_confidence = float(minimum_average_confidence)
        self._history = deque(maxlen=self.window_size)
        self.last_label = "none"
        self.stable_count = 0
        self.best_confidence = 0.0
        self._history.clear()

    def reset(self) -> None:
        self._history.clear()
        self.last_label = "none"
        self.stable_count = 0
        self.best_confidence = 0.0

    def select(self, detections: list[dict]) -> tuple[str, float]:
        bad = [d for d in detections if d.get("class_name") == self.bad_class_name]
        good = [d for d in detections if d.get("class_name") == self.good_class_name]
        if bad:
            selected = max(bad, key=lambda item: float(item.get("confidence", 0.0)))
            return self.bad_class_name, float(selected.get("confidence", 0.0))
        if good:
            selected = max(good, key=lambda item: float(item.get("confidence", 0.0)))
            return self.good_class_name, float(selected.get("confidence", 0.0))
        return "none", 0.0

    def update(self, detections: list[dict]) -> StableResult:
        label, confidence = self.select(detections)
        self._history.append((label, confidence))
        if label == self.last_label:
            self.stable_count += 1
            self.best_confidence = max(self.best_confidence, confidence)
        else:
            self.last_label = label
            self.stable_count = 1
            self.best_confidence = confidence

        matching = [value for item_label, value in self._history if item_label == label]
        votes = len(matching)
        average_confidence = sum(matching) / votes if votes else 0.0
        stable = (
            label != "none"
            and votes >= self.required_count
            and average_confidence >= self.minimum_average_confidence
        )
        return StableResult(label, max(matching, default=0.0), votes, stable)
