import cv2
import numpy as np
import pytest

from tire_vision.marker_detector import detect_yellow_marker
from tire_vision.tire_marker_test import DEFAULT_CONFIG, parse_args
from tire_vision.wear_measurement import (
    GeometryRecheckError,
    MarkerMissTracker,
    MeasurementError,
    classify_history,
    estimate_tire_geometry,
    estimate_tire_geometry_for_side,
    estimate_tire_geometry_from_bbox,
    measure_radial_gap,
)


def tire_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    return mask


def test_marker_miss_tracker_transition_boundaries():
    tracker = MarkerMissTracker(hold_frames=3, recheck_frames=5)
    assert tracker.marker_found("GOOD") == "GOOD"
    assert [tracker.marker_missing() for _ in range(6)] == [
        "GOOD", "GOOD", "GOOD", "RECHECK", "RECHECK", "ERROR"
    ]
    tracker.reset()
    assert tracker.marker_missing() == "RECHECK"


def test_circle_geometry():
    geometry = estimate_tire_geometry(tire_mask())
    assert geometry["center"] == pytest.approx((100, 100), abs=.2)
    assert geometry["radius"] == pytest.approx(60, abs=.2)


def test_yellow_marker_detection():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (94, 45), (106, 55), (0, 255, 255), -1)
    marker = detect_yellow_marker(frame, tire_mask(), (100, 100), 60)
    assert marker["center"] == pytest.approx((100, 50), abs=.5)
    assert marker["area"] == pytest.approx(120, abs=2)


def test_expanded_roi_detects_marker_just_outside_segmentation_mask():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    relaxed_yellow = cv2.cvtColor(
        np.uint8([[[20, 80, 100]]]), cv2.COLOR_HSV2BGR
    )[0, 0].tolist()
    cv2.rectangle(frame, (96, 34), (104, 39), relaxed_yellow, -1)
    with pytest.raises(MeasurementError, match="yellow marker not found"):
        detect_yellow_marker(
            frame, tire_mask(), (100, 100), 60, marker_roi_scale=1.0
        )
    marker = detect_yellow_marker(
        frame, tire_mask(), (100, 100), 60, marker_roi_scale=1.15
    )
    assert marker["area"] >= 5
    assert marker["center"][1] < 40


def test_radial_gap():
    contour = np.array([[[96, 45]], [[104, 45]], [[104, 55]], [[96, 55]]])
    measured = measure_radial_gap((100, 100), 60, contour, (100, 50))
    expected = np.linalg.norm(np.array([100, 40]) - np.array([96, 45]))
    assert measured["gap_px"] == pytest.approx(expected)
    assert measured["gap_ratio"] == pytest.approx(expected / 60)


@pytest.mark.parametrize("ratio", [.341, .367])
def test_good_result(ratio):
    assert classify_history([ratio] * 10, 10, .28, .32, .01)["result"] == "GOOD"


@pytest.mark.parametrize("ratio", [.18, .193])
def test_bad_result(ratio):
    assert classify_history([ratio] * 10, 10, .28, .32, .01)["result"] == "BAD"


@pytest.mark.parametrize("values", [[.341] * 9, [.30] * 10])
def test_recheck_result(values):
    assert classify_history(values, 10, .28, .32, .01)["result"] == "RECHECK"


def test_operating_threshold_boundaries_are_inclusive():
    assert classify_history([.28] * 10, 10, .28, .32, .01)["result"] == "BAD"
    assert classify_history([.32] * 10, 10, .28, .32, .01)["result"] == "GOOD"
    assert classify_history([.262] * 10, 10, .28, .32, .01)["result"] == "BAD"
    assert classify_history([.34] * 10, 10, .28, .32, .01)["result"] == "GOOD"


def test_high_standard_deviation_is_recheck():
    values = [.32, .36] * 5
    result = classify_history(values, 10, .28, .32, .01)
    assert result["std_ratio"] == pytest.approx(.02)
    assert result["result"] == "RECHECK"


def test_missing_marker_is_error():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(MeasurementError, match="yellow marker not found"):
        detect_yellow_marker(frame, tire_mask(), (100, 100), 60)


@pytest.mark.parametrize("mask", [
    np.zeros((100, 100), dtype=np.uint8),
    np.zeros((2, 2, 2), dtype=np.uint8),
])
def test_invalid_mask(mask):
    with pytest.raises(MeasurementError):
        estimate_tire_geometry(mask)


@pytest.mark.parametrize(
    ("side", "source", "bad_max", "good_min"),
    [
        ("left", "2", 0.28, 0.32),
        ("right", "4", 0.25, 0.27),
    ],
)
def test_standalone_side_selects_camera_and_thresholds(
    side, source, bad_max, good_min
):
    args = parse_args(["--config", str(DEFAULT_CONFIG), "--side", side])

    assert args.side == side
    assert args.source == source
    assert args.bad_max_ratio == bad_max
    assert args.good_min_ratio == good_min
    assert args.bad_max_gap_px == (None if side == "left" else 10.5)
    assert args.good_min_gap_px == (None if side == "left" else 11.5)
    assert args.max_ratio_std == (0.03 if side == "left" else 0.035)
    assert args.marker_hold_frames == 5
    assert args.marker_recheck_frames == 8


@pytest.mark.parametrize(
    ("source", "expected_side"),
    [("2", "left"), ("4", "right")],
)
def test_standalone_source_automatically_selects_side(source, expected_side):
    args = parse_args(["--config", str(DEFAULT_CONFIG), "--source", source])

    assert args.source == source
    assert args.side == expected_side


def test_right_bbox_geometry_keeps_only_largest_component():
    mask = np.zeros((220, 320), dtype=np.uint8)
    cv2.rectangle(mask, (50, 40), (149, 139), 255, -1)
    cv2.rectangle(mask, (280, 10), (289, 19), 255, -1)

    geometry = estimate_tire_geometry_from_bbox(mask)

    assert geometry["method"] == "bounding_rect"
    assert geometry["bbox"] == (50, 40, 150, 140)
    assert geometry["width"] == 100
    assert geometry["height"] == 100
    assert geometry["center"] == (100.0, 90.0)
    assert geometry["radius"] == 50.0
    assert geometry["mask"][15, 285] == 0
    assert geometry["mask"][90, 100] == 255


def test_right_bbox_geometry_rechecks_invalid_aspect_ratio():
    mask = np.zeros((200, 240), dtype=np.uint8)
    cv2.rectangle(mask, (30, 70), (169, 119), 255, -1)

    with pytest.raises(GeometryRecheckError, match="aspect ratio"):
        estimate_tire_geometry_for_side(mask, "right")


def test_left_geometry_dispatch_remains_min_enclosing_circle():
    direct = estimate_tire_geometry(tire_mask())
    dispatched = estimate_tire_geometry_for_side(tire_mask(), "left")

    assert dispatched["center"] == pytest.approx(direct["center"])
    assert dispatched["radius"] == pytest.approx(direct["radius"])
    assert "method" not in dispatched
