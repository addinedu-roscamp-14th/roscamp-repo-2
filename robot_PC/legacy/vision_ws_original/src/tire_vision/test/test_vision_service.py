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
