import os
import socket
import threading

import rclpy
import yaml
from ament_index_python.packages import (
    get_package_share_directory,
)
from rclpy.node import Node
from std_msgs.msg import String

from tire_arm_control.arm_driver import ArmDriver
from tire_arm_control.motion_player import (
    MotionPlayer,
    MotionStoppedError,
)


class ArmState:
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    HOMING = "HOMING"
    ERROR = "ERROR"


class TcpArmServerNode(Node):
    def __init__(self):
        super().__init__("tcp_arm_server_node")

        package_root = get_package_share_directory(
            "tire_arm_control"
        )
        config_path = os.path.join(
            package_root,
            "config",
            "arm_config.yaml",
        )

        with open(
            config_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.config = yaml.safe_load(f)

        self.arm = ArmDriver(
            port=self.config["robot_port"],
            baud=int(self.config["robot_baud"]),
        )

        motion_dir = os.path.join(
            package_root,
            self.config["motion_dir"],
        )
        self.player = MotionPlayer(
            self.arm,
            motion_dir,
        )

        self.command_pub = self.create_publisher(
            String,
            "/tire_arm_command_received",
            10,
        )

        self.status_pub = self.create_publisher(
            String,
            "/tire_arm_status",
            10,
        )

        self.tcp_host = self.config["tcp_host"]
        self.tcp_port = int(
            self.config["tcp_port"]
        )

        # 상태 및 모션 제어
        self.state_lock = threading.RLock()
        self.state = ArmState.IDLE
        self.current_command = ""
        self.current_motion = ""
        self.last_error = ""

        # 현재 모션의 남은 단계 취소에 사용
        self.stop_event = threading.Event()

        # 현재 모션 worker 종료 확인
        self.motion_done_event = threading.Event()
        self.motion_done_event.set()

        self.stop_timeout = float(
            self.config.get(
                "stop_timeout",
                3.0,
            )
        )
        self.stop_poll_interval = float(
            self.config.get(
                "stop_poll_interval",
                0.1,
            )
        )
        self.stop_stable_samples = int(
            self.config.get(
                "stop_stable_samples",
                3,
            )
        )
        self.stop_angle_tolerance = float(
            self.config.get(
                "stop_angle_tolerance",
                0.5,
            )
        )

        self.server_thread = threading.Thread(
            target=self.run_tcp_server,
            daemon=True,
        )
        self.server_thread.start()

        self.get_logger().info(
            "TCP arm server started on "
            f"{self.tcp_host}:{self.tcp_port}"
        )

    def publish_text(
        self,
        publisher,
        text,
    ):
        msg = String()
        msg.data = str(text)
        publisher.publish(msg)

    def set_state(
        self,
        state,
        *,
        command=None,
        motion=None,
        error=None,
    ):
        with self.state_lock:
            self.state = state

            if command is not None:
                self.current_command = command

            if motion is not None:
                self.current_motion = motion

            if error is not None:
                self.last_error = error

        self.publish_text(
            self.status_pub,
            f"STATE:{state}",
        )

        self.get_logger().info(
            f"Arm state changed to {state}"
        )

    def get_status_response(self):
        with self.state_lock:
            return (
                f"OK:STATUS"
                f":state={self.state}"
                f":command={self.current_command or '-'}"
                f":motion={self.current_motion or '-'}"
                f":error={self.last_error or '-'}"
            )

    @staticmethod
    def command_to_motion(command):
        command = command.strip().upper()

        if command == "OBSERVE_TIRE":
            return "observe_tire"

        if command == "REMOVE_BAD_TIRE":
            return "remove_bad_tire"

        if command == "HOME":
            return "home"

        if command == "HELP":
            return "help"

        if command == "SET_CAMERA":
            return "set_camera"

        if command == "SET_ARUCO":
            return "set_aruco"

        return None

    def is_motion_busy(self):
        with self.state_lock:
            return self.state in {
                ArmState.RUNNING,
                ArmState.HOMING,
                ArmState.STOP_REQUESTED,
                ArmState.STOPPING,
            }

    def handle_stop(self):
        """
        현재 모션을 선점 중단한다.

        응답은 mc.stop() 호출 직후가 아니라,
        모션 worker 종료와 실제 정지가 확인된 뒤 반환한다.
        """
        with self.state_lock:
            state = self.state

            if state in {
                ArmState.IDLE,
                ArmState.STOPPED,
            }:
                self.state = ArmState.STOPPED

                self.publish_text(
                    self.status_pub,
                    "STOPPED",
                )

                # 기존 TCP 응답 호환을 위해 OK:STOP 유지
                return "OK:STOP"

            if state == ArmState.ERROR:
                return (
                    "ERR:STATE_ERROR:"
                    f"{self.last_error or 'unknown'}"
                )

            self.state = ArmState.STOP_REQUESTED
            self.current_command = "STOP"

        self.publish_text(
            self.status_pub,
            "STOP_REQUESTED",
        )

        self.get_logger().warning(
            "STOP requested. Interrupting current motion."
        )

        # MotionPlayer가 남은 JSON 단계를 실행하지 못하게 한다.
        self.stop_event.set()

        self.set_state(
            ArmState.STOPPING,
            command="STOP",
        )

        try:
            # 현재 로봇 컨트롤러에 전달된 이동을 실제로 중단한다.
            self.arm.stop()
        except Exception as exc:
            error = f"Robot stop API failed: {exc}"

            self.set_state(
                ArmState.ERROR,
                error=error,
            )
            return f"ERR:STOP:{error}"

        # 모션 worker가 MotionStoppedError로 종료되기를 기다린다.
        worker_finished = self.motion_done_event.wait(
            timeout=self.stop_timeout
        )

        if not worker_finished:
            error = (
                "Motion worker did not stop within "
                f"{self.stop_timeout:.1f}s"
            )

            self.set_state(
                ArmState.ERROR,
                error=error,
            )
            return f"ERR:STOP_TIMEOUT:{error}"

        # 로봇이 실제로 움직이지 않는지 확인한다.
        stopped = self.arm.wait_until_stopped(
            timeout=self.stop_timeout,
            poll_interval=self.stop_poll_interval,
            stable_samples=self.stop_stable_samples,
            angle_tolerance=self.stop_angle_tolerance,
        )

        if not stopped:
            error = (
                "Robot movement did not stop within "
                f"{self.stop_timeout:.1f}s"
            )

            self.set_state(
                ArmState.ERROR,
                error=error,
            )
            return f"ERR:STOP_TIMEOUT:{error}"

        self.set_state(
            ArmState.STOPPED,
            command="STOP",
            motion="",
            error="",
        )

        self.publish_text(
            self.status_pub,
            "STOPPED",
        )

        self.get_logger().warning(
            "Robot motion stopped and verified."
        )

        # 중앙 PC 기존 성공 판정과 호환
        return "OK:STOP"

    def handle_tracking_command(self, command):
        parts = command.split()

        if len(parts) != 3:
            raise ValueError(
                "TRACK_JOINTS requires J1 and J3"
            )

        if self.is_motion_busy():
            with self.state_lock:
                current = self.current_command or self.state

            return f"ERR:BUSY:{current}"

        self.arm.send_j1_j3(
            float(parts[1]),
            float(parts[2]),
            int(
                self.config.get(
                    "tracking_speed",
                    30,
                )
            ),
        )

        self.publish_text(
            self.status_pub,
            "DONE:TRACK_JOINTS",
        )

        return "OK:TRACK_JOINTS"

    def handle_focus_servo(self):
        if self.is_motion_busy():
            with self.state_lock:
                current = self.current_command or self.state

            return f"ERR:BUSY:{current}"

        self.arm.focus_servos()

        self.publish_text(
            self.status_pub,
            "DONE:FOCUS_SERVO",
        )

        return "OK:FOCUS_SERVO"

    def play_motion(self, command, motion_name):
        """
        호출한 클라이언트 스레드에서 모션을 실행한다.

        accept 루프와 분리되어 있으므로 이 함수가 블로킹되어도
        서버는 STOP을 포함한 새 TCP 연결을 계속 받을 수 있다.
        """
        with self.state_lock:
            if self.state in {
                ArmState.RUNNING,
                ArmState.HOMING,
                ArmState.STOP_REQUESTED,
                ArmState.STOPPING,
            }:
                current = (
                    self.current_command
                    or self.state
                )
                return f"ERR:BUSY:{current}"

            if (
                self.state == ArmState.ERROR
                and command != "HOME"
            ):
                return (
                    "ERR:STATE_ERROR:"
                    f"{self.last_error or 'unknown'}"
                )

            if (
                self.state == ArmState.STOPPED
                and command != "HOME"
            ):
                return "ERR:STOPPED:HOME_REQUIRED"

            self.stop_event.clear()
            self.motion_done_event.clear()
            self.current_command = command
            self.current_motion = motion_name
            self.last_error = ""

            if command == "HOME":
                self.state = ArmState.HOMING
            else:
                self.state = ArmState.RUNNING

        self.publish_text(
            self.status_pub,
            f"PLAYING:{motion_name}",
        )

        self.get_logger().info(
            f"Playing motion: {motion_name}"
        )

        try:
            self.player.play(
                motion_name,
                stop_event=self.stop_event,
            )

        except MotionStoppedError as exc:
            self.get_logger().warning(
                f"Motion interrupted: {exc}"
            )

            # STOP 처리 스레드가 최종 STOPPED 상태를 결정한다.
            with self.state_lock:
                if self.state not in {
                    ArmState.STOP_REQUESTED,
                    ArmState.STOPPING,
                    ArmState.STOPPED,
                    ArmState.ERROR,
                }:
                    self.state = ArmState.STOPPED

            self.publish_text(
                self.status_pub,
                f"INTERRUPTED:{motion_name}",
            )

            return f"ERR:STOPPED:{motion_name}"

        except Exception as exc:
            error = str(exc)

            self.set_state(
                ArmState.ERROR,
                error=error,
            )

            self.get_logger().error(
                f"Motion failed: {error}"
            )

            return f"ERR:{error}"

        else:
            with self.state_lock:
                stop_requested = (
                    self.stop_event.is_set()
                    or self.state
                    in {
                        ArmState.STOP_REQUESTED,
                        ArmState.STOPPING,
                        ArmState.STOPPED,
                    }
                )

                if not stop_requested:
                    self.state = ArmState.IDLE
                    self.current_command = ""
                    self.current_motion = ""
                    self.last_error = ""

            if stop_requested:
                self.publish_text(
                    self.status_pub,
                    f"INTERRUPTED:{motion_name}",
                )
                return f"ERR:STOPPED:{motion_name}"

            self.publish_text(
                self.status_pub,
                f"DONE:{motion_name}",
            )

            return f"OK:{motion_name}"

        finally:
            self.motion_done_event.set()

    def handle_command(self, command):
        command = command.strip()
        normalized = command.upper()

        self.get_logger().info(
            f"Received command: {normalized}"
        )

        self.publish_text(
            self.command_pub,
            normalized,
        )

        # STOP과 STATUS는 어떤 상태에서도 우선 처리한다.
        if normalized == "STOP":
            return self.handle_stop()

        if normalized == "STATUS":
            return self.get_status_response()

        if normalized.startswith("TRACK_JOINTS"):
            return self.handle_tracking_command(
                normalized
            )

        if normalized == "FOCUS_SERVO":
            return self.handle_focus_servo()

        motion_name = self.command_to_motion(
            normalized
        )

        if motion_name is None:
            raise ValueError(
                f"Unknown command: {normalized}"
            )

        return self.play_motion(
            normalized,
            motion_name,
        )

    def handle_client(self, conn, addr):
        """
        각 TCP 연결을 별도 스레드에서 처리한다.

        한 클라이언트가 긴 모션 응답을 기다리는 동안에도
        accept 루프는 다음 STOP 연결을 받을 수 있다.
        """
        try:
            with conn:
                conn.settimeout(10.0)

                data = conn.recv(1024)

                if not data:
                    conn.sendall(b"ERR:EMPTY\n")
                    return

                command = data.decode(
                    "utf-8"
                ).strip()

                if not command:
                    conn.sendall(b"ERR:EMPTY\n")
                    return

                try:
                    response = self.handle_command(
                        command
                    )
                except Exception as exc:
                    response = f"ERR:{exc}"
                    self.get_logger().error(
                        response
                    )

                conn.sendall(
                    (response + "\n").encode(
                        "utf-8"
                    )
                )

        except socket.timeout:
            self.get_logger().error(
                f"Client timeout: {addr}"
            )

        except Exception as exc:
            self.get_logger().error(
                f"Client error {addr}: {exc}"
            )

    def run_tcp_server(self):
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as server:
            server.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )

            server.bind(
                (
                    self.tcp_host,
                    self.tcp_port,
                )
            )
            server.listen(10)
            server.settimeout(1.0)

            while rclpy.ok():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True,
                )
                client_thread.start()


def main(args=None):
    rclpy.init(args=args)
    node = TcpArmServerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()