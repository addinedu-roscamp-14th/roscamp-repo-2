from tire_vision.tire_detector import (
    MockTireDetector,
    YoloTireDetector,
    create_detector,
)
from tire_vision.wear_measurement import (
    GeometryRecheckError,
    MarkerNotFoundError,
    MeasurementError,
)


def make_detector(min_frames=2):
    return YoloTireDetector(
        model_path="unused-in-unit-test.pt",
        model=object(),
        ratio_window_size=3,
        ratio_min_valid_frames=min_frames,
        bad_max_ratio=.28,
        good_min_ratio=.32,
        max_ratio_std=.01,
    )


def measured(ratio):
    return (
        {"confidence": .9, "bbox": [1.0, 2.0, 30.0, 40.0]},
        {"gap_ratio": ratio},
    )


def test_operating_marker_defaults():
    detector = YoloTireDetector(
        model_path="unused-in-unit-test.pt", model=object()
    )
    stabilizer = detector._ratio_stabilizers["left"]
    assert detector.yellow_lower == (15, 40, 60)
    assert detector.yellow_upper == (42, 255, 255)
    assert detector.min_marker_area == 5
    assert detector.marker_roi_scale == 1.15
    assert stabilizer.window_size == 15
    assert stabilizer.min_valid_frames == 10
    assert stabilizer.max_ratio_std == .02


def test_returns_legacy_bad_label_after_ratio_stabilizes():
    detector = make_detector()
    detector._measure_frame = lambda frame, side=None: measured(.18)
    assert detector.detect(None, side="left") == []
    assert detector.detect(None, side="left") == [{
        "class_id": 0,
        "class_name": "bad_tire",
        "confidence": .9,
        "bbox": [1.0, 2.0, 30.0, 40.0],
    }]


def test_returns_legacy_good_label_after_ratio_stabilizes():
    detector = make_detector()
    detector._measure_frame = lambda frame, side=None: measured(.341)
    assert detector.detect(None, side="right") == []
    assert detector.detect(None, side="right")[0]["class_name"] == "good_tire"


def test_recheck_and_measurement_error_return_no_detections():
    detector = make_detector()
    detector._measure_frame = lambda frame, side=None: measured(.30)
    assert detector.detect(None, side="left") == []
    assert detector.detect(None, side="left") == []

    def fail(frame, side=None):
        raise MarkerNotFoundError("yellow marker not found")

    detector._measure_frame = fail
    assert detector.detect(None, side="left") == []
    assert detector.status_for_side("left") == "RECHECK"


def test_marker_miss_holds_then_rechecks_then_errors():
    detector = make_detector(min_frames=1)
    detector._measure_frame = lambda frame, side=None: measured(.431)
    assert detector.detect(None, side="left")[0]["class_name"] == "good_tire"

    def miss(frame, side=None):
        raise MarkerNotFoundError("yellow marker not found")

    detector._measure_frame = miss
    for expected_count in (1, 2, 3):
        held = detector.detect(None, side="left")
        assert held[0]["class_name"] == "good_tire"
        assert detector.status_for_side("left") == "GOOD"
        assert detector.marker_diagnostics("left")["miss_count"] == expected_count
    for expected_count in (4, 5):
        assert detector.detect(None, side="left") == []
        assert detector.status_for_side("left") == "RECHECK"
        assert detector.marker_diagnostics("left")["miss_count"] == expected_count
    assert detector.detect(None, side="left") == []
    assert detector.status_for_side("left") == "ERROR"
    assert detector.marker_diagnostics("left")["miss_count"] == 6


def test_left_and_right_ratio_histories_are_independent():
    detector = make_detector()
    ratios = iter([.18, .341, .18, .341])
    detector._measure_frame = lambda frame, side=None: measured(next(ratios))
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
    detector._measure_frame = lambda frame, side=None: measured(.18)
    detector.detect(None, side="left")
    detector.detect(None, side="right")
    assert len(detector._ratio_stabilizers["left"].history) == 1
    assert len(detector._ratio_stabilizers["right"].history) == 1

    detector.reset_histories()

    assert len(detector._ratio_stabilizers["left"].history) == 0
    assert len(detector._ratio_stabilizers["right"].history) == 0
    assert detector.status_for_side("left") == "NO_TIRE"
    assert detector.status_for_side("right") == "NO_TIRE"


def test_left_and_right_use_different_decision_thresholds():
    detector = YoloTireDetector(
        model_path="unused-in-unit-test.pt",
        model=object(),
        ratio_min_valid_frames=1,
        side_decision_args={
            "left": {"bad_max_ratio": 0.28, "good_min_ratio": 0.32},
            "right": {"bad_max_ratio": 0.30, "good_min_ratio": 0.35},
        },
    )
    detector._measure_frame = lambda frame, side=None: measured(
        0.32 if side == "left" else 0.30
    )

    assert detector.detect(None, side="left")[0]["class_name"] == "good_tire"
    assert detector.detect(None, side="right")[0]["class_name"] == "bad_tire"


def test_create_detector_passes_complete_camera_decision_config(monkeypatch):
    captured = {}

    def fake_yolo_detector(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "tire_vision.tire_detector.YoloTireDetector", fake_yolo_detector
    )
    create_detector({
        "mock_mode": False,
        "model_path": "unused.pt",
        "left_camera": {
            "bad_max_ratio": 0.28,
            "good_min_ratio": 0.32,
            "max_ratio_std": 0.03,
            "marker_hold_frames": 5,
            "marker_recheck_frames": 8,
            "window_size": 15,
            "min_valid_frames": 10,
        },
        "right_camera": {
            "bad_max_gap_px": 10.5,
            "good_min_gap_px": 11.5,
            "bad_max_ratio": 0.25,
            "good_min_ratio": 0.27,
            "max_ratio_std": 0.035,
            "marker_hold_frames": 5,
            "marker_recheck_frames": 8,
            "window_size": 15,
            "min_valid_frames": 10,
        },
    })

    left = captured["side_decision_args"]["left"]
    right = captured["side_decision_args"]["right"]
    assert left["bad_max_ratio"] == 0.28
    assert left["good_min_ratio"] == 0.32
    assert left["max_ratio_std"] == 0.03
    assert left["marker_hold_frames"] == 5
    assert left["marker_recheck_frames"] == 8
    assert right["bad_max_gap_px"] == 10.5
    assert right["good_min_gap_px"] == 11.5
    assert right["bad_max_ratio"] == 0.25
    assert right["good_min_ratio"] == 0.27
    assert right["max_ratio_std"] == 0.035
    assert right["marker_hold_frames"] == 5
    assert right["marker_recheck_frames"] == 8


def test_right_geometry_aspect_failure_is_recheck_without_valid_frame():
    detector = make_detector()

    def invalid_geometry(frame, side=None):
        raise GeometryRecheckError("right tire bbox aspect ratio out of range")

    detector._measure_frame = invalid_geometry
    assert detector.detect(None, side="right") == []

    state = detector.decision_state("right")
    assert state["result"] == "RECHECK"
    assert state["error_code"] == "GEOMETRY_RECHECK"
    assert state["valid_count"] == 0
    assert state["marker_miss_count"] == 0
