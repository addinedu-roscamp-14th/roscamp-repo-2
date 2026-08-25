import json
import pytest

from tire_vision.protocol import ProtocolError, encode_response, parse_request


def test_parse_request_accepts_json_line():
    assert parse_request(b'{"type":"health"}\n') == {"type": "health"}


def test_parse_request_rejects_invalid_json():
    with pytest.raises(ProtocolError) as error:
        parse_request(b"{bad json}\n")
    assert error.value.code == "INVALID_JSON"


def test_encode_response_is_newline_delimited_json():
    assert encode_response({"status": "OK"}).endswith(b"\n")


def test_protocol_encode_decode_round_trip():
    encoded = encode_response(
        {"version": 1, "type": "health", "request_id": "round-trip", "status": "ok"}
    )
    decoded = json.loads(encoded)
    assert decoded["request_id"] == "round-trip"
    assert decoded["status"] == "ok"
