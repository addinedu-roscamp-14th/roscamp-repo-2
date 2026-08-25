"""Hardware-independent interface for future transport ArUco services/actions."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Protocol

from central_control.replacement_policy import validate_transport_slot


@dataclass(frozen=True)
class TransportOperationResult:
    success: bool
    message: str = ""


class TransportOperationHandler(Protocol):
    def place_removed_tire(self, side: str, slot: int) -> TransportOperationResult:
        ...

    def pick_new_tire(self, side: str, slot: int) -> TransportOperationResult:
        ...


class MockTransportOperationHandler:
    """Bounded fake used by central orchestration tests; it never contacts hardware."""

    def __init__(self, fail_operations=None):
        self.fail_operations = set(fail_operations or [])
        self.calls: list[tuple[str, str, int]] = []

    def _result(self, operation: str, side: str, tire_kind: str, slot: int):
        slot = validate_transport_slot(side, tire_kind, slot)
        self.calls.append((operation, side, slot))
        failed = operation in self.fail_operations or (operation, side) in self.fail_operations
        return TransportOperationResult(
            not failed,
            f"MOCK {operation} side={side} slot={slot}",
        )

    def place_removed_tire(self, side: str, slot: int) -> TransportOperationResult:
        return self._result("PLACE_REMOVED_TIRE", side, "removed_tire", slot)

    def pick_new_tire(self, side: str, slot: int) -> TransportOperationResult:
        return self._result("PICK_NEW_TIRE", side, "new_tire", slot)



class RosTransportOperationHandler:
    """Synchronous worker-thread adapter around bounded ROS service calls."""

    OPERATIONS = {
        "PLACE_REMOVED_TIRE": "removed_tire",
        "PICK_NEW_TIRE": "new_tire",
    }

    def __init__(
        self,
        node,
        service_type,
        service_names: dict[str, str],
        timeout_sec: float,
        service_wait_timeout_sec: float = 2.0,
    ):
        self.service_type = service_type
        self.timeout_sec = max(0.001, float(timeout_sec))
        self.service_wait_timeout_sec = max(
            0.001, float(service_wait_timeout_sec)
        )
        expected_sides = {"left", "right"}
        if set(service_names) != expected_sides:
            raise ValueError("transport service names must define left and right")
        self.service_names = {
            side: str(service_names[side]) for side in sorted(expected_sides)
        }
        self.clients = {
            side: node.create_client(service_type, self.service_names[side])
            for side in sorted(expected_sides)
        }

    def _call(self, operation: str, side: str, slot: int):
        tire_kind = self.OPERATIONS[operation]
        slot = validate_transport_slot(side, tire_kind, slot)
        client = self.clients[side]
        started_at = time.monotonic()
        wait_timeout = min(self.service_wait_timeout_sec, self.timeout_sec)
        if not client.wait_for_service(timeout_sec=wait_timeout):
            return TransportOperationResult(
                False,
                f"{operation} service unavailable: {self.service_names[side]}",
            )

        request = self.service_type.Request()
        request.operation = operation
        request.slot = slot
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        remaining = max(0.0, self.timeout_sec - (time.monotonic() - started_at))
        if not completed.wait(remaining):
            return TransportOperationResult(
                False,
                f"{operation} timeout after {self.timeout_sec:.3f}s "
                f"side={side} slot={slot}",
            )
        try:
            response = future.result()
        except Exception as error:
            return TransportOperationResult(
                False,
                f"{operation} service error side={side} slot={slot}: {error}",
            )
        if response is None:
            return TransportOperationResult(
                False,
                f"{operation} returned no response side={side} slot={slot}",
            )
        echoed = (
            str(response.side),
            str(response.operation),
            int(response.slot),
        )
        expected = (side, operation, slot)
        if echoed != expected:
            return TransportOperationResult(
                False,
                f"transport response mismatch: expected={expected} actual={echoed}",
            )
        return TransportOperationResult(bool(response.success), str(response.message))

    def place_removed_tire(self, side: str, slot: int) -> TransportOperationResult:
        return self._call("PLACE_REMOVED_TIRE", side, slot)

    def pick_new_tire(self, side: str, slot: int) -> TransportOperationResult:
        return self._call("PICK_NEW_TIRE", side, slot)


class UnavailableTransportOperationHandler:
    """Fail closed until a real ROS service/action adapter is explicitly injected."""

    def place_removed_tire(self, side: str, slot: int) -> TransportOperationResult:
        return TransportOperationResult(False, "PLACE_REMOVED_TIRE handler unavailable")

    def pick_new_tire(self, side: str, slot: int) -> TransportOperationResult:
        return TransportOperationResult(False, "PICK_NEW_TIRE handler unavailable")

