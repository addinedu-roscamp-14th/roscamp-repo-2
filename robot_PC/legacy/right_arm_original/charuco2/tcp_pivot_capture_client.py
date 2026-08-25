import json
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROBOT_IP = "192.168.5.1"
ROBOT_PORT = 5000

DATASET_DIR = Path(
    "/home/choiminjoon/tcp_pivot_dataset"
)
SAMPLES_FILE = DATASET_DIR / "tcp_pivot_samples.json"

MIN_CAPTURE_INTERVAL_SEC = 1.0


class RobotClient:
    def __init__(
        self,
        ip: str,
        port: int,
    ) -> None:
        self.sock = socket.create_connection(
            (ip, port),
            timeout=5.0,
        )
        self.sock.settimeout(20.0)
        self.sock.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )
        self.buffer = ""

    def request(
        self,
        command: str,
    ) -> Dict[str, Any]:
        self.sock.sendall(
            (
                command.strip()
                + "\n"
            ).encode("utf-8")
        )

        while "\n" not in self.buffer:
            chunk = self.sock.recv(4096)

            if not chunk:
                raise ConnectionError(
                    "JetCobot 서버 연결이 종료되었습니다."
                )

            self.buffer += chunk.decode(
                "utf-8",
                errors="ignore",
            )

        line, self.buffer = self.buffer.split(
            "\n",
            1,
        )

        return json.loads(line)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def load_samples() -> List[Dict[str, Any]]:
    if not SAMPLES_FILE.exists():
        return []

    data = json.loads(
        SAMPLES_FILE.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(data, list):
        raise RuntimeError(
            "tcp_pivot_samples.json 형식이 잘못되었습니다."
        )

    return data


def save_samples(
    samples: List[Dict[str, Any]],
) -> None:
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLES_FILE.write_text(
        json.dumps(
            samples,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_samples(
    samples: List[Dict[str, Any]],
) -> None:
    print()
    print("-" * 62)
    print(f"저장 샘플 수: {len(samples)}")

    for sample in samples:
        index = sample["index"]
        coords = sample["coords_mm_deg"]
        print(
            f"{index:02d}: "
            f"XYZ=({coords[0]:.1f}, "
            f"{coords[1]:.1f}, "
            f"{coords[2]:.1f}) mm | "
            f"RPY=({coords[3]:.1f}, "
            f"{coords[4]:.1f}, "
            f"{coords[5]:.1f}) deg"
        )

    print("-" * 62)


def main() -> None:
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = load_samples()

    robot = RobotClient(
        ROBOT_IP,
        ROBOT_PORT,
    )

    print("서버 응답:", robot.request("PING"))
    print("서버 상태:", robot.request("STATUS"))

    print()
    print("=" * 62)
    print("TCP 피벗 보정 자세 수집")
    print("f : 서보 해제 — 반드시 로봇팔을 손으로 받치기")
    print("e : 서보 고정")
    print("c : 현재 자세 저장")
    print("u : 마지막 샘플 삭제")
    print("l : 샘플 목록")
    print("q : 서보 고정 후 종료")
    print()
    print("권장 샘플: 8~12개")
    print("포인터 끝은 모든 자세에서 같은 고정점에 유지")
    print("=" * 62)

    print_samples(samples)

    last_capture_time = 0.0

    try:
        while True:
            command = input(
                "\n명령 [f/e/c/u/l/q]: "
            ).strip().lower()

            if command == "f":
                print()
                print(
                    "주의: 로봇팔을 손으로 단단히 받치세요."
                )

                confirm = input(
                    "서보를 해제할까요? [y/N]: "
                ).strip().lower()

                if confirm == "y":
                    print(
                        "RELEASE:",
                        robot.request("RELEASE"),
                    )
                else:
                    print("취소")

            elif command == "e":
                print(
                    "LOCK:",
                    robot.request("LOCK"),
                )
                time.sleep(1.0)

            elif command == "c":
                now = time.monotonic()

                if (
                    now - last_capture_time
                    < MIN_CAPTURE_INTERVAL_SEC
                ):
                    print(
                        "잠시 기다린 뒤 다시 저장하세요."
                    )
                    continue

                response = robot.request("POSE")

                if not response.get("ok"):
                    print(
                        "POSE 읽기 실패:",
                        response,
                    )
                    continue

                sample = {
                    "index": len(samples) + 1,
                    "timestamp": (
                        datetime.now().isoformat(
                            timespec="milliseconds"
                        )
                    ),
                    "coords_mm_deg": [
                        float(value)
                        for value in response[
                            "coords_mm_deg"
                        ]
                    ],
                    "angles_deg": [
                        float(value)
                        for value in response[
                            "angles_deg"
                        ]
                    ],
                }

                samples.append(sample)
                save_samples(samples)
                last_capture_time = now

                print(
                    f"샘플 {sample['index']} 저장 완료"
                )
                print(
                    "coords:",
                    sample["coords_mm_deg"],
                )

            elif command == "u":
                if not samples:
                    print("삭제할 샘플이 없습니다.")
                    continue

                removed = samples.pop()

                for index, sample in enumerate(
                    samples,
                    start=1,
                ):
                    sample["index"] = index

                save_samples(samples)

                print(
                    "삭제:",
                    removed,
                )

            elif command == "l":
                print_samples(samples)

            elif command == "q":
                print(
                    "LOCK:",
                    robot.request("LOCK"),
                )
                print_samples(samples)
                break

            else:
                print(
                    "f, e, c, u, l, q 중에서 입력하세요."
                )

    except KeyboardInterrupt:
        print("\nCtrl+C 입력")

    finally:
        try:
            print(
                "종료 전 LOCK:",
                robot.request("LOCK"),
            )
        except Exception as exc:
            print("LOCK 실패:", exc)

        robot.close()
        print("수집기 종료")


if __name__ == "__main__":
    main()