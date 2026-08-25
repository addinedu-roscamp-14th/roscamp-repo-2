import socket


# =========================================================
# 오른쪽 JetCobot 서버
# =========================================================
ROBOT_IP = "192.168.5.1"
ROBOT_PORT = 5000


# =========================================================
# 키별 명령
# =========================================================
COMMANDS = {
    "0": "HOME",

    "1": "TEST,X,5",
    "2": "TEST,Y,5",
    "3": "TEST,Z,5",

    "4": "TEST,X,-5",
    "5": "TEST,Y,-5",
    "6": "TEST,Z,-5",

    "s": "STOP"
}


def receive_line(sock):
    """서버 응답 한 줄을 받는다."""
    received = b""

    while b"\n" not in received:
        data = sock.recv(4096)

        if not data:
            raise ConnectionError(
                "서버 연결이 종료되었습니다."
            )

        received += data

    line = received.split(
        b"\n",
        1
    )[0]

    return line.decode(
        "utf-8",
        errors="ignore"
    ).strip()


# =========================================================
# 서버 연결
# =========================================================
client = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

client.setsockopt(
    socket.IPPROTO_TCP,
    socket.TCP_NODELAY,
    1
)

client.settimeout(5.0)


print(
    f"오른팔 서버 연결 시도: "
    f"{ROBOT_IP}:{ROBOT_PORT}"
)

try:
    client.connect(
        (ROBOT_IP, ROBOT_PORT)
    )

except socket.timeout:
    print("서버 연결 시간 초과")
    client.close()
    raise SystemExit

except ConnectionRefusedError:
    print("서버 연결 거부됨")
    print(
        "JetCobot에서 "
        "right_robot_home_test_server.py를 "
        "먼저 실행하세요."
    )
    client.close()
    raise SystemExit

except OSError as error:
    print("서버 연결 실패:", error)
    client.close()
    raise SystemExit


# 이동 완료 응답을 기다릴 수 있도록 시간 연장
client.settimeout(20.0)

print("오른팔 서버 연결 완료")


# =========================================================
# 조작 안내
# =========================================================
print()
print("========================================")
print("HOME 기준 Cartesian 축 시험")
print("========================================")
print("0 : HOME 복귀")
print()
print("1 : HOME 기준 X +5mm")
print("2 : HOME 기준 Y +5mm")
print("3 : HOME 기준 Z +5mm")
print()
print("4 : HOME 기준 X -5mm")
print("5 : HOME 기준 Y -5mm")
print("6 : HOME 기준 Z -5mm")
print()
print("s : 즉시 STOP")
print("q : STOP 후 종료")
print("========================================")
print()
print(
    "각 축 시험 후 반드시 0을 눌러 "
    "HOME으로 복귀하세요."
)


try:
    while True:
        key = input(
            "\n명령 > "
        ).strip().lower()

        # -------------------------------------------------
        # 종료
        # -------------------------------------------------
        if key == "q":
            try:
                client.sendall(
                    b"STOP\n"
                )

                response = receive_line(
                    client
                )

                print(
                    "서버 응답:",
                    response
                )

            except OSError:
                pass

            break

        # -------------------------------------------------
        # 명령 확인
        # -------------------------------------------------
        command = COMMANDS.get(key)

        if command is None:
            print("잘못된 입력입니다.")
            continue

        # -------------------------------------------------
        # 명령 전송
        # -------------------------------------------------
        try:
            client.sendall(
                f"{command}\n".encode(
                    "utf-8"
                )
            )

            print("전송:", command)
            print("로봇 이동 완료 대기 중...")

            response = receive_line(
                client
            )

            print(
                "서버 응답:",
                response
            )

        except socket.timeout:
            print(
                "서버 응답 시간 초과 → STOP 전송"
            )

            try:
                client.sendall(
                    b"STOP\n"
                )
            except OSError:
                pass

        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionError,
            OSError
        ) as error:
            print(
                "서버 통신 오류:",
                error
            )
            break


except KeyboardInterrupt:
    print("\nCtrl+C 입력됨")

    try:
        client.sendall(
            b"STOP\n"
        )
    except OSError:
        pass


finally:
    try:
        client.sendall(
            b"STOP\n"
        )
    except OSError:
        pass

    try:
        client.shutdown(
            socket.SHUT_RDWR
        )
    except OSError:
        pass

    client.close()

    print("축 시험 클라이언트 종료")