#!/usr/bin/env python3
"""
스테이션 리스트를 순서대로 순회하는 웨이포인트 매니저.

  1. Nav2 NavigateToPose로 approach point까지 이동 (장애물 회피, loose tolerance)
     - 단, 정밀 정차가 필요한 스테이션은 approach_pose까지 CAPTURE_RADIUS(10cm) 안으로
       들어오는 순간 Nav2 목표를 취소하고 곧바로 도킹 상태 머신으로 넘어감.
       Nav2의 목표 근처 감속/재가속(부자연스러운 속도 변화, 부정확한 도착)을 피하고
       더 이른 지점부터 도킹 컨트롤러의 예측 가능한 접근으로 이어받기 위함.
  2. 도킹 상태 머신(dock_control.py)이 ROTATE/MOVE 로직으로 후진 + 정밀 yaw 정렬 수행
     (Nav2는 이 구간 동안 관여하지 않음)
  3. 도킹 완료 후 WAIT(타이어 교체 대기)
  4. 다음 스테이션으로 반복, 마지막 스테이션 이후 처음으로 돌아가 루프(한 바퀴)

[장애물 정지-재개 추가]
도킹 구간(2번)은 Nav2 관할 밖이라 원래 라이다를 전혀 안 봤음. /scan을 구독해서
도킹 중 현재 이동 방향(dock_control.py가 알려줌) 기준 콘 안의 최소거리를 계산,
DockingStateMachine.step()에 넘겨서 막히면 정지-재개하도록 함.

[라이다 마운트 오프셋 보정]
tf_static 확인 결과 rplidar_link가 base_link 기준 z축 180도 회전되어 마운트됨
(rplidar_mount->rplidar_link 쿼터니언 z=1, w≈0). 즉 라이다 좌표계 0도가 로봇
후방을 가리킴. get_move_direction_relative()가 주는 "로봇 기준 각도"를 라이다
좌표계로 변환할 때 LIDAR_YAW_OFFSET(180도)을 더해줘야 함. 이걸 빠뜨렸을 때
로봇 정면 장애물을 계속 못 잡고 그대로 밀고 가는 문제가 실측으로 확인됨.

TODO:
  - is_wait_condition을 고정시간 대신 Jetcobot 완료 신호(토픽/서비스)로 교체할 경우
    WAITING 분기의 조건문만 바꾸면 됨
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from nav2_msgs.action import NavigateToPose

from pinky_goal_pid.dock_control import DockingStateMachine
import numpy as np


# goal_pid.py에서 쓰던 값 그대로. 필요시 rqt_reconfigure로 붙여서 튜닝해도 됨.
DOCKING_PARAMS = {
    'kp_angle': 0.6,
    'kp_dist': 0.3,
    'kd_dist': 0.05,
    'max_lin': 0.08,
    'max_ang': 0.29,
    'goal_tolerance': 0.01,
    'yaw_tolerance': 0.015,
    'min_ang_vel': 0.1,
    'min_lin_vel': 0.008,
    'move_yaw_correction_gain': 0.3,
    'move_yaw_correction_limit': 0.0087,
    'move_yaw_deadzone_deg': 0.5,
    'move_start_yaw_threshold_deg': 2.0,
    'kp_angle_diagonal': 0.5,
    'max_ang_diagonal': 0.2,
    # 상태 전환(MOVE<->ROTATE, Nav2->도킹 등) 시 속도 급변 방지용 변화율 제한.
    # max_lin=0.08 기준 0→최대속도 약 0.4초, max_ang=0.29 기준 약 0.3초에 도달하도록 설정.
    # 너무 느리게 느껴지면 값을 키우고, 여전히 튀는 느낌이면 낮춰서 튜닝.
    'max_lin_accel': 0.2,   # m/s^2
    'max_ang_accel': 1.0,   # rad/s^2
    # 도킹 중 이동 방향 전방 최소거리가 이보다 가까우면 정지 (튜닝 필요, 실측 후 조정)
    'docking_obstacle_stop_dist': 0.15,
}

# 도킹 중 장애물 체크용 라이다 콘 반각(rad). 이동 방향 기준 ±이 값 안에서 최소거리 계산.
# 튜닝 필요: 로봇 폭/속도에 비해 너무 좁으면 옆으로 살짝 비껴간 장애물을 못 잡고,
# 너무 넓으면 진행 방향과 무관한 물체에도 과민 반응함.
OBSTACLE_CONE_HALF_ANGLE = math.radians(25)

# rplidar_link가 base_link 기준 z축 180도 회전되어 마운트됨 (tf_static으로 확인).
# 로봇 기준 각도 -> 라이다 좌표계 각도 변환 시 이 값을 더해야 함.
LIDAR_YAW_OFFSET = math.pi

# Nav2 목표 전송 관련 타이밍
STARTUP_DELAY_SEC = 3.0     # 노드 시작 후 첫 목표 전송까지 대기 (Nav2 lifecycle 안정화 시간)
GOAL_RETRY_DELAY_SEC = 1.0  # 목표 거부/서버 미응답 시 재시도 간격

# Nav2 주행 중 approach_pose까지 이 거리(m) 안으로 들어오면 도킹으로 조기 전환.
# 정밀 정차가 필요한 스테이션(docking_waypoints가 있는 경우)에만 적용됨.
CAPTURE_RADIUS = 0.01


class Station:
    def __init__(self, name, approach_pose, docking_waypoints=None, wait_seconds=0.0):
        """
        approach_pose: (x, y, yaw) - Nav2 목표. general_goal_checker(loose tolerance) 적용.
        docking_waypoints: 정밀 정차가 필요한 스테이션만 채움 (타이어 운반 정차 지점).
                            None이면 approach 도착 후 바로 다음 스테이션으로 넘어감.
                            goal_pid.py 포맷과 동일: (x, y, yaw, move_type, [wait_seconds])
        wait_seconds: 도킹 완료 후 대기시간(타이어 교체 등). docking_waypoints가 없으면 무시됨.
        """
        self.name = name
        self.approach_pose = approach_pose
        self.docking_waypoints = docking_waypoints
        self.wait_seconds = wait_seconds


class WaypointManager(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        self.sub_pose = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.pose_callback, qos)
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.latest_scan = None
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        self.nav2_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.current_x = None
        self.current_y = None
        self.current_yaw = None
        self.current_lin_vel = 0.0
        self.current_ang_vel = 0.0

        self.stations = [
            Station('approach_tire_stop_1', approach_pose=(1.508, 0.335, 3.138),
                    docking_waypoints=[
                        (1.508, 0.335, -1.572, 'ROTATE'),
                        (1.508, 0.475, -1.572, 'MOVE_BACKWARD'),
                    ],
                    wait_seconds=3.0),
            Station('approach_tire_stop_2', approach_pose=(1.508, 0.475, 3.138),
                    docking_waypoints=[
                        (1.508, 0.475, 3.135, 'ROTATE'),
                        (1.085, 0.475, 3.135, 'MOVE_FORWARD'),
                    ],
                    wait_seconds=3.0),
            Station('approach_tire_stop_3', approach_pose=(0.492, 0.433, -2.631),
                    docking_waypoints=[
                        (0.300, 0.458, 3.135, 'MOVE_FORWARD'),
                    ],
                    wait_seconds=3.0),
            Station('approach_tire_stop_4', approach_pose=(0.303, 0.473, 3.138),
                    docking_waypoints=[
                        (0.143, 0.473, 3.135, 'MOVE_FORWARD'),
                        (0.143, 0.465, -1.565, 'ROTATE'),
                        (0.143, 0.035, -1.565, 'MOVE_FORWARD'),
                    ],
                    wait_seconds=0.0),
            # Station('approach_tire_stop_test', approach_pose=(0.143, 0.035, -2.831),
                    # docking_waypoints=[
                        # (0.143, 0.035, -1.566, 'MOVE_FORWARD'),
                    # ],
                    # wait_seconds=3.0),
            Station('return_to_start_via', approach_pose=(1.11, 0.18, 0.222), docking_waypoints=None, wait_seconds=0.0),

            Station('return_to_start', approach_pose=(1.592, 0.061, 0.222),
                    docking_waypoints=[
                        (1.572, 0.081, -3.138, 'MOVE_DIAGONAL'),
                        (1.572, 0.081, -3.138, 'ROTATE'),
                    ],
                    wait_seconds=0.0),
        ]

        self.station_index = 0
        self.docking_sm = None
        self.mode = 'NAV2'  # 'NAV2' -> 'DOCKING' -> 'WAITING'
        self.wait_start_time = None

        # Nav2 조기 취소/전환 관련 상태
        self.current_goal_handle = None
        self.docking_triggered_early = False

        # 재시도/재전송용 1회성 타이머 핸들. 여러 개 쌓이지 않도록 항상 취소 후 재생성.
        self._retry_timer = None

        self.declare_parameter('loop_mode', True)
        self.loop_mode = self.get_parameter('loop_mode').value

        self.timer = self.create_timer(0.1, self.control_loop)   # 20Hz -> 10Hz, CPU 부담 완화

        # Nav2 lifecycle(bt_navigator 등)이 active 상태로 안정화될 시간을 벌어준 뒤 시작.
        # wait_for_server는 액션 서버 '존재'만 확인하지 '목표 처리 준비'는 보장 안 하므로,
        # 초반에 바로 목표를 보내면 accept/reject 스팸이 발생할 수 있음.
        self.get_logger().info(f'{STARTUP_DELAY_SEC}초 후 첫 목표 전송...')
        self._retry_timer = self.create_timer(STARTUP_DELAY_SEC, self._start_once)

    def _start_once(self):
        self._cancel_retry_timer()
        self.send_next_nav2_goal()

    def _cancel_retry_timer(self):
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    def _schedule_retry(self, delay_sec, callback):
        self._cancel_retry_timer()
        self._retry_timer = self.create_timer(delay_sec, lambda: self._fire_once(callback))

    def _fire_once(self, callback):
        self._cancel_retry_timer()
        callback()

    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_yaw = yaw

    def odom_callback(self, msg):
        self.current_lin_vel = msg.twist.twist.linear.x
        self.current_ang_vel = msg.twist.twist.angular.z

    def scan_callback(self, msg):
        self.latest_scan = msg

    @staticmethod
    def _min_range_in_cone(scan, center_angle, half_angle):
        """스캔에서 center_angle(라이다 좌표계 기준, rad) ± half_angle 콘 안의
        최소 유효거리 반환. 유효 range 없으면 None.
        numpy 벡터화 버전 (기존 파이썬 for문 대비 CPU 부담 완화, load average
        과부하 확인 후 적용)."""
        ranges = np.asarray(scan.ranges)
        n = len(ranges)
        if n == 0:
            return None
        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        valid = (ranges >= scan.range_min) & (ranges <= scan.range_max)
        diff = np.arctan2(np.sin(angles - center_angle), np.cos(angles - center_angle))
        in_cone = valid & (np.abs(diff) <= half_angle)

        if not np.any(in_cone):
            return None
        return float(np.min(ranges[in_cone]))

    # ---------- Nav2 구간 ----------
    def send_next_nav2_goal(self):
        station = self.stations[self.station_index]
        x, y, yaw = station.approach_pose

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.get_logger().info(
            f'[{station.name}] Nav2 목표 전송: ({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}도)')
        self.mode = 'NAV2'
        self.current_goal_handle = None
        self.docking_triggered_early = False

        if not self.nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f'Nav2 액션 서버 응답 없음, {GOAL_RETRY_DELAY_SEC}초 후 재시도')
            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self.send_next_nav2_goal)
            return

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.nav2_goal_response_callback)

    def nav2_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(
                f'Nav2 목표 거부됨, {GOAL_RETRY_DELAY_SEC}초 후 재시도')
            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self.send_next_nav2_goal)
            return
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav2_result_callback)

    def nav2_result_callback(self, future):
        # 이미 capture radius 진입으로 도킹으로 조기 전환된 경우, 뒤늦게 도착하는
        # (취소됐거나 정상 완료된) Nav2 결과 콜백은 무시한다. 안 그러면 이미 DOCKING
        # 모드로 넘어간 뒤에 advance_to_next_station이 다시 불려서 스테이션이 씹힐 수 있음.
        if self.docking_triggered_early:
            return

        station = self.stations[self.station_index]
        self.get_logger().info(f'[{station.name}] Nav2 도착 완료')

        if station.docking_waypoints:
            self.get_logger().info(f'[{station.name}] 정밀 도킹 시작')
            self.docking_sm = DockingStateMachine(
                station.docking_waypoints, DOCKING_PARAMS, self.get_logger(), self.get_clock())
            self.mode = 'DOCKING'
        else:
            self.advance_to_next_station()

    def _trigger_early_docking(self):
        """NAV2 주행 중 approach_pose까지 CAPTURE_RADIUS 안에 들어왔을 때 호출.
        Nav2 목표를 취소하고 즉시 도킹 상태 머신으로 전환한다."""
        self.docking_triggered_early = True

        if self.current_goal_handle is not None:
            self.current_goal_handle.cancel_goal_async()

        station = self.stations[self.station_index]
        self.get_logger().info(
            f'[{station.name}] Nav2 주행 중 {CAPTURE_RADIUS:.2f}m 이내 진입, '
            f'도킹으로 조기 전환')
        self.docking_sm = DockingStateMachine(
            station.docking_waypoints, DOCKING_PARAMS, self.get_logger(), self.get_clock())
        self.mode = 'DOCKING'

    # ---------- 도킹/대기/전환 구간 ----------
    def advance_to_next_station(self):
        self.station_index += 1
        if self.station_index >= len(self.stations):
            if self.loop_mode:
                self.get_logger().info('한 바퀴 완료 -> 루프 재시작')
                self.station_index = 0
            else:
                self.get_logger().info('전체 경로 완료')
                self.mode = 'DONE'
                return
        self.send_next_nav2_goal()

    def control_loop(self):
        if self.current_x is None:
            return

        if self.mode == 'NAV2':
            station = self.stations[self.station_index]
            # 정밀 정차가 필요한 스테이션만 조기 전환 로직 적용
            if station.docking_waypoints and not self.docking_triggered_early:
                ax, ay, _ = station.approach_pose
                dist = math.hypot(ax - self.current_x, ay - self.current_y)
                if dist < CAPTURE_RADIUS:
                    self._trigger_early_docking()
            # Nav2 controller_server가 /cmd_vel을 직접 발행 중이므로 여기서는 개입하지 않음
            return

        elif self.mode == 'DOCKING':
            min_obstacle_dist = None
            if self.latest_scan is not None:
                rel_angle = self.docking_sm.get_move_direction_relative(
                    self.current_x, self.current_y, self.current_yaw)
                if rel_angle is not None:
                    # 로봇 기준 각도(rel_angle)를 라이다 좌표계 각도로 변환.
                    # rplidar_link가 base_link 대비 180도 회전 마운트되어 있어서
                    # LIDAR_YAW_OFFSET(pi)을 더한 뒤 -pi~pi로 정규화한다.
                    laser_angle = math.atan2(
                        math.sin(rel_angle + LIDAR_YAW_OFFSET),
                        math.cos(rel_angle + LIDAR_YAW_OFFSET))
                    min_obstacle_dist = self._min_range_in_cone(
                        self.latest_scan, laser_angle, OBSTACLE_CONE_HALF_ANGLE)
                    # self.get_logger().info(
                        # f'[DEBUG] rel_angle={math.degrees(rel_angle):.1f}도 '
                        # f'laser_angle={math.degrees(laser_angle):.1f}도 '
                        # f'min_dist={min_obstacle_dist}')

            cmd, done = self.docking_sm.step(
                self.current_x, self.current_y, self.current_yaw,
                self.current_lin_vel, self.current_ang_vel,
                min_obstacle_dist=min_obstacle_dist)
            self.cmd_pub.publish(cmd)
            if done:
                self.cmd_pub.publish(Twist())
                station = self.stations[self.station_index]
                self.get_logger().info(f'[{station.name}] 도킹 완료')
                if station.wait_seconds > 0:
                    self.wait_start_time = self.get_clock().now()
                    self.mode = 'WAITING'
                else:
                    self.advance_to_next_station()

        elif self.mode == 'WAITING':
            self.cmd_pub.publish(Twist())
            station = self.stations[self.station_index]
            elapsed = (self.get_clock().now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= station.wait_seconds:
                self.get_logger().info(f'[{station.name}] 대기 종료')
                self.advance_to_next_station()


def main():
    rclpy.init()
    node = WaypointManager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()