import cv2
import numpy as np
import pytest

from tire_vision.marker_detector import detect_yellow_marker
from tire_vision.wear_measurement import (
    MeasurementError,
    classify_history,
    estimate_tire_geometry,
    measure_radial_gap,
)


def tire_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    return mask


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


def test_radial_gap():
    contour = np.array([[[96, 45]], [[104, 45]], [[104, 55]], [[96, 55]]])
    measured = measure_radial_gap((100, 100), 60, contour, (100, 50))
    expected = np.linalg.norm(np.array([100, 40]) - np.array([96, 45]))
    assert measured["gap_px"] == pytest.approx(expected)
    assert measured["gap_ratio"] == pytest.approx(expected / 60)


@pytest.mark.parametrize("ratio", [.341, .367])
def test_good_result(ratio):
    assert classify_history([ratio] * 10, 10, .23, .30, .01)["result"] == "GOOD"


@pytest.mark.parametrize("ratio", [.18, .193])
def test_bad_result(ratio):
    assert classify_history([ratio] * 10, 10, .23, .30, .01)["result"] == "BAD"


@pytest.mark.parametrize("values", [[.341] * 9, [.26] * 10])
def test_recheck_result(values):
    assert classify_history(values, 10, .23, .30, .01)["result"] == "RECHECK"


def test_high_standard_deviation_is_recheck():
    values = [.32, .36] * 5
    result = classify_history(values, 10, .23, .30, .01)
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
