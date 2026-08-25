import json
import socket
from typing import Any, Dict


ROBOT_IP = "192.168.5.1"
ROBOT_PORT = 5000


def request(
    sock: socket.socket,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    sock.sendall(
        (
            json.dumps(payload, ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    )

    buffer = ""

    while "\n" not in buffer:
        data = sock.recv(4096)

        if not data:
            raise ConnectionError(
                "로봇 서버 연결이 종료되었습니다."
            )

        buffer += data.decode(
            "utf-8",
            errors="ignore",
        )

    line, _ = buffer.split("\n", 1)
    return json.loads(line)


def main() -> None:
    print("그리퍼 주변에 손과 물체가 없는지 확인하세요.")

    with socket.create_connection(
        (ROBOT_IP, ROBOT_PORT),
        timeout=5.0,
    ) as sock:
        sock.settimeout(20.0)

        print("서버 확인:", request(
            sock,
            {"command": "PING"},
        ))

        while True:
            command = input(
                "o=열기, c=닫기, q=종료: "
            ).strip().lower()

            if command == "o":
                print(
                    "열기 결과:",
                    request(
                        sock,
                        {"command": "GRIPPER_OPEN"},
                    ),
                )

            elif command == "c":
                print(
                    "닫기 결과:",
                    request(
                        sock,
                        {"command": "GRIPPER_CLOSE"},
                    ),
                )

            elif command == "q":
                print("종료합니다.")
                break

            else:
                print("o, c, q 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()