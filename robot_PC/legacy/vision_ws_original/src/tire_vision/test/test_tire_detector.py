from tire_vision.tire_detector import MockTireDetector, YoloTireDetector
from tire_vision.wear_measurement import MeasurementError


def make_detector(min_frames=2):
    return YoloTireDetector(
        model_path="unused-in-unit-test.pt",
        model=object(),
        ratio_window_size=3,
        ratio_min_valid_frames=min_frames,
        bad_max_ratio=.23,
        good_min_ratio=.30,
        max_ratio_std=.01,
    )


def measured(ratio):
    return (
        {"confidence": .9, "bbox": [1.0, 2.0, 30.0, 40.0]},
        {"gap_ratio": ratio},
    )


def test_returns_legacy_bad_label_after_ratio_stabilizes():
    detector = make_detector()
    detector._measure_frame = lambda frame: measured(.18)
    assert detector.detect(None, side="left") == []
    assert detector.detect(None, side="left") == [{
        "class_id": 0,
        "class_name": "bad_tire",
        "confidence": .9,
        "bbox": [1.0, 2.0, 30.0, 40.0],
    }]


def test_returns_legacy_good_label_after_ratio_stabilizes():
    detector = make_detector()
    detector._measure_frame = lambda frame: measured(.341)
    assert detector.detect(None, side="right") == []
    assert detector.detect(None, side="right")[0]["class_name"] == "good_tire"


def test_recheck_and_measurement_error_return_no_detections():
    detector = make_detector()
    detector._measure_frame = lambda frame: measured(.26)
    assert detector.detect(None, side="left") == []
    assert detector.detect(None, side="left") == []

    def fail(frame):
        raise MeasurementError("yellow marker not found")

    detector._measure_frame = fail
    assert detector.detect(None, side="left") == []


def test_left_and_right_ratio_histories_are_independent():
    detector = make_detector()
    ratios = iter([.18, .341, .18, .341])
    detector._measure_frame = lambda frame: measured(next(ratios))
    assert detector.detect(None, side="left") == []
    assert detector.detect(None, side="right") == []
    assert detector.detect(None, side="left")[0]["class_name"] == "bad_tire"
    assert detector.detect(None, side="right")[0]["class_name"] == "good_tire"


def test_mock_detector_behavior_is_unchanged():
    detector = MockTireDetector()
    assert detector.detect(None, side="left")[0]["class_name"] == "bad_tire"
    assert detector.detect(None, side="right")[0]["class_name"] == "good_tire"


def test_reset_histories_clears_left_and_right_ratio_cycles():
    detector = make_detector()
    detector._measure_frame = lambda frame: measured(.18)
    detector.detect(None, side="left")
    detector.detect(None, side="right")
    assert len(detector._ratio_stabilizers["left"].history) == 1
    assert len(detector._ratio_stabilizers["right"].history) == 1

    detector.reset_histories()

    assert len(detector._ratio_stabilizers["left"].history) == 0
    assert len(detector._ratio_stabilizers["right"].history) == 0
    assert detector.status_for_side("left") == "RECHECK"
    assert detector.status_for_side("right") == "RECHECK"
