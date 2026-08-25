from tire_vision.detection_stabilizer import DetectionStabilizer


def test_bad_tire_has_priority_over_good_tire():
    stabilizer = DetectionStabilizer(required_count=1)
    result = stabilizer.update(
        [
            {"class_name": "good_tire", "confidence": 0.99},
            {"class_name": "bad_tire", "confidence": 0.5},
        ]
    )
    assert result.label == "bad_tire"
    assert result.stable


def test_requires_repeated_non_none_label():
    stabilizer = DetectionStabilizer(required_count=2)
    first = stabilizer.update([{"class_name": "bad_tire", "confidence": 0.7}])
    second = stabilizer.update([{"class_name": "bad_tire", "confidence": 0.8}])
    assert not first.stable
    assert second.stable
    assert second.confidence == 0.8


def test_minimum_average_confidence_blocks_unreliable_result():
    stabilizer = DetectionStabilizer(
        required_count=2, window_size=3, minimum_average_confidence=0.8
    )
    stabilizer.update([{"class_name": "good_tire", "confidence": 0.7}])
    result = stabilizer.update([{"class_name": "good_tire", "confidence": 0.8}])
    assert result.stable_count == 2
    assert not result.stable


def test_reset_clears_vote_history():
    stabilizer = DetectionStabilizer(required_count=2)
    stabilizer.update([{"class_name": "bad_tire", "confidence": 0.9}])
    stabilizer.reset()
    result = stabilizer.update([{"class_name": "bad_tire", "confidence": 0.9}])
    assert result.stable_count == 1
    assert not result.stable
