import json
import socket
import threading

from right_arm_transport_pick.robot_client import RightArmRobotClient


def test_client_uses_server_commands_and_fresh_connections():
    requests = []
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(4)

    def serve():
        for _ in range(4):
            conn, _ = server.accept()
            with conn:
                request = json.loads(conn.recv(4096).split(b"\n", 1)[0])
                requests.append(request)
                if request["command"] == "GET_COORDS":
                    body = {"status": "ok", "data": {"coords": [1, 2, 3, 4, 5, 6]}}
                elif request["command"] == "MOVE_COORDS":
                    body = {
                        "status": "ok",
                        "data": {
                            "reached": True,
                            "final_coords_mm_deg": request["coords"],
                        },
                    }
                else:
                    body = {"status": "ok", "data": {}}
                conn.sendall((json.dumps(body) + "\n").encode())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = RightArmRobotClient("127.0.0.1", server.getsockname()[1])
    assert client.ping() is True
    client.move_single_coord(2, 99.0)
    client.gripper_open()
    thread.join(timeout=2.0)
    server.close()
    assert [item["command"] for item in requests] == ["GET_COORDS", "GET_COORDS", "MOVE_COORDS", "GRIPPER_OPEN"]
    assert requests[2]["coords"] == [1, 99.0, 3, 4, 5, 6]
    assert all(item["version"] == 1 and item["request_id"] for item in requests)
