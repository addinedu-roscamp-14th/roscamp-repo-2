import threading
import time

from tire_vision.camera_manager import FakeCamera
from tire_vision.vision_service import VisionService


class SlowCameraManager:
    def capture_frame(self):
        time.sleep(0.2)
        return FakeCamera().read()[1]


class MockDetector:
    def detect(self, frame):
        return [{"class_name": "bad_tire", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]


class MissingMarkerDetector:
    def detect(self, frame, side=None):
        return []

    def status_for_side(self, side):
        return "RECHECK"


class ResetTrackingDetector(MockDetector):
    def __init__(self):
        self.reset_calls = 0

    def reset_histories(self):
        self.reset_calls += 1


def test_health_works_in_mock_mode_without_weights():
    service = VisionService({"mock_mode": True})
    response = service.handle_request({"type": "health"})
    assert response["status"] == "ok"
    assert response["health"]["mock_mode"] is True
    assert response["health"]["model_available"] is False
    assert response["health"]["left_camera_available"] is True
    assert response["health"]["right_camera_available"] is True


def test_health_preserves_protocol_fields():
    service = VisionService({"mock_mode": True})
    response = service.handle_request(
        {"version": 1, "type": "health", "request_id": "health-001"}
    )
    assert response["version"] == 1
    assert response["request_id"] == "health-001"
    assert response["status"] == "ok"


def test_detect_tires_uses_mock_camera_and_detector():
    service = VisionService(
        {"mock_mode": True, "required_stable_count": 1},
        detector=MockDetector(),
    )
    response = service.handle_request({"type": "detect_tires", "side": "right"})
    assert response["status"] == "ok"
    assert response["side"] == "right"
    assert response["result"] == "bad_tire"


def test_reset_cycle_clears_detector_and_both_service_stabilizers():
    detector = ResetTrackingDetector()
    service = VisionService(
        {"mock_mode": True, "required_stable_count": 1},
        detector=detector,
    )
    service._stabilizers["left"].update([
        {"class_name": "bad_tire", "confidence": .9}
    ])
    service._stabilizers["right"].update([
        {"class_name": "good_tire", "confidence": .9}
    ])
    response = service.handle_request({
        "type": "detect_tires",
        "reset_cycle": True,
    })
    assert detector.reset_calls == 1
    assert response["left"]["frames"] == 1
    assert response["right"]["frames"] == 1
    assert response["left"]["status"] == "BAD"
    assert response["right"]["status"] == "BAD"


def test_zero_valid_marker_frames_returns_recheck_not_error():
    service = VisionService(
        {
            "mock_mode": True,
            "required_stable_count": 1,
            "detect_timeout_sec": 0.02,
            "detect_loop_sleep_sec": 0.001,
        },
        detector=MissingMarkerDetector(),
    )
    response = service.handle_request({"type": "detect_tires", "side": "left"})
    assert response["status"] == "ok"
    assert response["detection_status"] == "RECHECK"
    assert response["result"] == "none"
    assert response["stable"] is False


def test_concurrent_detection_returns_busy():
    service = VisionService(
        {"mock_mode": True, "required_stable_count": 1},
        camera_manager=SlowCameraManager(),
        detector=MockDetector(),
    )
    first = {}

    def run_first():
        first.update(service.handle_request({"type": "detect_tires"}))

    thread = threading.Thread(target=run_first)
    thread.start()
    time.sleep(0.05)
    second = service.handle_request({"type": "detect_tires"})
    thread.join()

    assert first["status"] == "ok"
    assert second == {
        "status": "error",
        "version": 1,
        "type": "error",
        "request_id": None,
        "error_code": "BUSY",
        "message": "Detection is already running",
    }



def test_detect_tires_returns_left_and_right_by_default():
    service = VisionService({"mock_mode": True, "required_stable_count": 1})
    response = service.handle_request(
        {"version": 1, "type": "detect_tires", "request_id": "detect-001"}
    )

    assert response["version"] == 1
    assert response["type"] == "detection_result"
    assert response["request_id"] == "detect-001"
    assert response["status"] == "ok"
    assert response["left"]["class_name"] == "bad_tire"
    assert response["left"]["confidence"] == 0.91
    assert response["left"]["stable"] is True
    assert response["left"]["votes"] == 1
    assert response["left"]["frames"] == 1
    assert response["left"]["detections"] == []
    assert response["right"]["class_name"] == "good_tire"
    assert response["right"]["confidence"] == 0.88
    assert response["right"]["stable"] is True
    assert response["right"]["votes"] == 1
    assert response["right"]["frames"] == 1
    assert response["right"]["detections"] == []


def test_default_mock_response_has_three_votes_and_frames():
    service = VisionService({"mock_mode": True, "required_stable_count": 3})
    response = service.handle_request(
        {"version": 1, "type": "detect_tires", "request_id": "mock-001"}
    )
    assert response["left"]["votes"] == response["left"]["frames"] == 3
    assert response["right"]["votes"] == response["right"]["frames"] == 3
    assert response["left"]["confidence"] == 0.91
    assert response["right"]["confidence"] == 0.88


def test_unknown_request_type_is_rejected():
    import pytest
    from tire_vision.protocol import ProtocolError

    service = VisionService({"mock_mode": True})
    with pytest.raises(ProtocolError) as error:
        service.handle_request({"type": "invalid"})
    assert error.value.code == "UNKNOWN_REQUEST"


def test_busy_response_preserves_request_id():
    service = VisionService(
        {"mock_mode": True, "required_stable_count": 1},
        camera_manager=SlowCameraManager(),
        detector=MockDetector(),
    )
    thread = threading.Thread(
        target=lambda: service.handle_request(
            {"version": 1, "type": "detect_tires", "request_id": "first"}
        )
    )
    thread.start()
    time.sleep(0.05)
    second = service.handle_request(
        {"version": 1, "type": "detect_tires", "request_id": "second"}
    )
    thread.join()

    assert second["status"] == "error"
    assert second["type"] == "error"
    assert second["request_id"] == "second"
    assert second["error_code"] == "BUSY"


class RatioStateDetector:
    uses_decision_state = True

    def __init__(self):
        self.reset_calls = 0
        self.counts = {"left": 0, "right": 0}

    def reset_sessions(self):
        self.reset_calls += 1
        self.counts = {"left": 0, "right": 0}

    def detect(self, frame, side=None):
        self.counts[side] += 1
        if self.counts[side] < 2:
            return []
        return [{
            "class_name": "good_tire",
            "confidence": 0.93,
            "bbox": [0, 0, 1, 1],
        }]

    def decision_state(self, side):
        count = self.counts[side]
        return {
            "result": "GOOD" if count >= 2 else "COLLECTING",
            "valid_count": count,
            "median_ratio": 0.431 if count else None,
            "std_ratio": 0.0 if count else None,
            "session_id": self.reset_calls,
            "error_code": None,
            "detail": "",
        }

    def status_for_side(self, side):
        return self.decision_state(side)["result"]


def test_ratio_state_detector_does_not_require_second_vote_window():
    detector = RatioStateDetector()
    service = VisionService(
        {"mock_mode": True, "required_stable_count": 3},
        detector=detector,
    )

    first = service.handle_request({"type": "detect_tires"})
    second = service.handle_request({"type": "detect_tires"})

    assert detector.reset_calls == 2
    for response in (first, second):
        assert response["detection_status"] == "OK"
        assert response["left"]["result"] == "GOOD"
        assert response["right"]["result"] == "GOOD"
        assert response["left"]["frames"] == 2
        assert response["right"]["frames"] == 2
        assert response["left"]["votes"] == 2
        assert response["right"]["votes"] == 2


def paired_side_result(status, session_id):
    return {
        "status": status,
        "result": status,
        "session_id": session_id,
        "error_code": None,
        "message": "",
    }


def test_one_side_recheck_discards_both_and_retries_paired_detection():
    detector = ResetTrackingDetector()
    service = VisionService(
        {"mock_mode": True, "max_detection_retries": 3},
        detector=detector,
    )

    def detect_side(side, request):
        status = "RECHECK" if detector.reset_calls == 1 and side == "right" else "GOOD"
        return paired_side_result(status, detector.reset_calls)

    service._detect_side = detect_side
    response = service.handle_request({"type": "detect_tires"})

    assert response["detection_status"] == "OK"
    assert response["retry_count"] == 1
    assert response["attempts"] == 2
    assert detector.reset_calls == 2
    assert response["left"]["session_id"] == 2
    assert response["right"]["session_id"] == 2


def test_paired_recheck_retry_limit_returns_detection_error():
    detector = ResetTrackingDetector()
    service = VisionService(
        {"mock_mode": True, "max_detection_retries": 2},
        detector=detector,
    )
    service._detect_side = lambda side, request: paired_side_result(
        "RECHECK", detector.reset_calls
    )

    response = service.handle_request({"type": "detect_tires"})

    assert response["detection_status"] == "ERROR"
    assert response["error_code"] == "DETECTION_ERROR"
    assert response["retry_count"] == 2
    assert response["attempts"] == 3
    assert detector.reset_calls == 3


class ReconnectingCameraManager:
    def __init__(self):
        self.counts = {"left": 0, "right": 0}

    def reconnect_count(self, side):
        return self.counts[side]

    def capture_frame(self, side):
        self.counts[side] += 1
        return FakeCamera().read()[1]


def test_camera_reconnect_resets_both_detection_states():
    detector = ResetTrackingDetector()
    camera_manager = ReconnectingCameraManager()
    service = VisionService(
        {"mock_mode": True},
        camera_manager=camera_manager,
        detector=detector,
    )

    service._capture_frame("left")

    assert detector.reset_calls == 1
    assert service._camera_reconnected_during_cycle is True
    assert service._stabilizers["left"].stable_count == 0
    assert service._stabilizers["right"].stable_count == 0
