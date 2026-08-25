import json
from typing import Any


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def parse_request(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    line = line.strip()
    if not line:
        raise ProtocolError("EMPTY_REQUEST", "request line is empty")

    try:
        request = json.loads(line)
    except json.JSONDecodeError as error:
        raise ProtocolError("INVALID_JSON", str(error)) from error

    if not isinstance(request, dict):
        raise ProtocolError("INVALID_REQUEST", "request must be a JSON object")
    if not isinstance(request.get("type"), str):
        raise ProtocolError("INVALID_REQUEST", "request.type is required")
    return request


def request_id_from(request: dict[str, Any] | None) -> str | None:
    if not request:
        return None
    request_id = request.get("request_id")
    return str(request_id) if request_id is not None else None


def version_from(request: dict[str, Any] | None) -> int:
    if not request:
        return 1
    try:
        return int(request.get("version", 1))
    except (TypeError, ValueError):
        return 1


def make_response(status: str, **fields: Any) -> dict[str, Any]:
    response = {"status": status.lower()}
    response.update(fields)
    return response


def make_error(
    code: str,
    message: str,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return make_response(
        "error",
        version=version_from(request),
        type="error",
        request_id=request_id_from(request),
        error_code=code,
        message=message,
    )


def encode_response(response: dict[str, Any]) -> bytes:
    return (json.dumps(response, separators=(",", ":"), sort_keys=True) + "\n").encode(
        "utf-8"
    )

