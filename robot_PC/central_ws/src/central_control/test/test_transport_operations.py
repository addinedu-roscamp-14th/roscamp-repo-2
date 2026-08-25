from types import SimpleNamespace

import pytest

from central_control.transport_operations import RosTransportOperationHandler


class TransportTask:
    class Request:
        def __init__(self):
            self.operation = ""
            self.slot = 0


class FakeFuture:
    def __init__(self, response=None, complete=True):
        self.response = response
        self.complete = complete

    def add_done_callback(self, callback):
        if self.complete:
            callback(self)

    def result(self):
        return self.response


class FakeClient:
    def __init__(self, side, calls, complete=True):
        self.side = side
        self.calls = calls
        self.complete = complete

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        self.calls.append((self.side, request.operation, request.slot))
        response = SimpleNamespace(
            success=True,
            message="OK",
            side=self.side,
            operation=request.operation,
            slot=request.slot,
        )
        return FakeFuture(response, complete=self.complete)


class FakeNode:
    def __init__(self, complete=True):
        self.complete = complete
        self.calls = []

    def create_client(self, _service_type, service_name):
        side = "left" if service_name.startswith("/left_arm/") else "right"
        return FakeClient(side, self.calls, complete=self.complete)


def make_handler(node, timeout_sec=0.1):
    return RosTransportOperationHandler(
        node,
        TransportTask,
        {
            "left": "/left_arm/transport_task",
            "right": "/right_arm/transport_task",
        },
        timeout_sec=timeout_sec,
        service_wait_timeout_sec=0.01,
    )


@pytest.mark.parametrize(
    "method,side,slot,operation",
    [
        ("place_removed_tire", "left", 3, "PLACE_REMOVED_TIRE"),
        ("pick_new_tire", "left", 4, "PICK_NEW_TIRE"),
        ("place_removed_tire", "right", 1, "PLACE_REMOVED_TIRE"),
        ("pick_new_tire", "right", 2, "PICK_NEW_TIRE"),
    ],
)
def test_ros_transport_routes_side_operation_and_slot(method, side, slot, operation):
    node = FakeNode()
    result = getattr(make_handler(node), method)(side, slot)
    assert result.success
    assert node.calls == [(side, operation, slot)]


def test_ros_transport_timeout_calls_service_once():
    node = FakeNode(complete=False)
    result = make_handler(node, timeout_sec=0.01).place_removed_tire("left", 3)
    assert not result.success
    assert "timeout" in result.message
    assert node.calls == [("left", "PLACE_REMOVED_TIRE", 3)]


@pytest.mark.parametrize(
    "method,side,slot",
    [
        ("place_removed_tire", "left", 1),
        ("pick_new_tire", "left", 2),
        ("place_removed_tire", "right", 3),
        ("pick_new_tire", "right", 4),
    ],
)
def test_ros_transport_rejects_wrong_slot_before_service(method, side, slot):
    node = FakeNode()
    with pytest.raises(ValueError):
        getattr(make_handler(node), method)(side, slot)
    assert node.calls == []

