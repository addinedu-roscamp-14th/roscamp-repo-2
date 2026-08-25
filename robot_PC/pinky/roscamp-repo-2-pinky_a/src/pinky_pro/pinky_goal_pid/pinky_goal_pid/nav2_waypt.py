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

[Nav2 목표 실패 처리 추가]
기존엔 nav2_result_callback이 goal_handle의 결과 status를 확인하지 않고
콜백이 오기만 하면 무조건 "도착 완료"로 처리했음. 그래서 planner 실패/충돌 등으로
실제로는 approach_pose에 도달 못 했는데도 도킹 상태 머신이 그 자리를 기준으로
상대 이동을 시작해버려 좌표가 틀어지는 문제가 있었음.
이제 result.status를 확인해서 STATUS_SUCCEEDED가 아니면 도킹으로 넘어가지 않고
NAV2_MAX_RETRIES 횟수까지 같은 목표를 재시도하고, 그래도 안 되면 ERROR 모드로
정지함 (무한 재시도로 코스트맵이 구조적으로 막힌 경우에 계속 실패만 반복하는 것 방지).

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
from action_msgs.msg import GoalStatus

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
    'min_lin_vel': 0.01,
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
GOAL_RETRY_DELAY_SEC = 1.0  # 목표 거부/서버 미응답/실패 시 재시도 간격

# Nav2 주행 중 approach_pose까지 이 거리(m) 안으로 들어오면 도킹으로 조기 전환.
# 정밀 정차가 필요한 스테이션(docking_waypoints가 있는 경우)에만 적용됨.
CAPTURE_RADIUS = 0.01

# Nav2 목표 실패(planning 실패, collision 등) 시 재시도 최대 횟수.
# 이 횟수 넘으면 재시도 포기하고 ERROR 모드로 정지.
NAV2_MAX_RETRIES = 3


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
    def __init__(self, node_name='waypoint_manager'):
        super().__init__(node_name)

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
            Station('approach_tire_stop_1', approach_pose=(0.2233, -0.4619, -0.9535),
                docking_waypoints=[
                    (0.2145, -0.4537, -0.0006, 'ROTATE'),
                    (0.4975, -0.4612, -0.0212, 'MOVE_FORWARD'),
                ],
                wait_seconds=0.0),
            Station('approach_tire_stop_2', approach_pose=(0.8927, -0.413, -0.0012),
                docking_waypoints=[
                    (1.3248, -0.4169, -0.0044, 'MOVE_FORWARD'),
                ],
                wait_seconds=0.0),
            Station('approach_tire_stop_3', approach_pose=(1.324, -0.416, -0.0044),  # 비워둠, 추후 측정 필요
                docking_waypoints=[
                    (1.4749, -0.4215, -0.0187, 'MOVE_FORWARD'),
                    (1.4781, -0.4166, 1.5289, 'ROTATE'),
                    (1.5047, 0.0240, 1.5551, 'MOVE_FORWARD'),
                    (1.4980, 0.0293, 3.1184, 'ROTATE'),
                    (0.9738, 0.0289, 3.1231, 'MOVE_FORWARD'),
                ],
                wait_seconds=0.0),
            Station('return_to_start_via', approach_pose=(0.4574, -0.1194, -2.8615), docking_waypoints=None, wait_seconds=0.0),

            Station('return_to_start', approach_pose=(0.1381, -0.0245, 1.966),  # 기존 값 유지
                docking_waypoints=[
                    (0.1381, -0.0245, 1.966, 'MOVE_DIAGONAL'),
                    (0.1588, -0.0213, 0.0059, 'ROTATE'),
                    (-0.0249, -0.0221, -0.0043, 'MOVE_BACKWARD'),
                ],
                wait_seconds=0.0),
        ]

        self.station_index = 0
        self.docking_sm = None
        self.mode = 'IDLE'  # 외부 START_ROUTE 전에는 절대로 목표를 보내지 않는다.
        self.route_state = 'IDLE'
        self.route_name = None
        self.route_error = ''
        self.route_end_index = len(self.stations)
        self.wait_start_time = None

        # Nav2 조기 취소/전환 관련 상태
        self.current_goal_handle = None
        self.docking_triggered_early = False
        self._goal_seq = 0
        self._nav2_fail_count = 0   # 현재 스테이션에 대한 연속 Nav2 실패 횟수

        # 재시도/재전송용 1회성 타이머 핸들. 여러 개 쌓이지 않도록 항상 취소 후 재생성.
        self._retry_timer = None

        self.declare_parameter('loop_mode', False)
        self.loop_mode = self.get_parameter('loop_mode').value

        self.timer = self.create_timer(0.1, self.control_loop)   # 20Hz -> 10Hz, CPU 부담 완화
        self.get_logger().info('경로 매니저 준비 완료: IDLE (START_ROUTE 대기)')

    def start_route(self, route='after_tire_service'):
        """Start one central-control route segment.

        tire_stop_1: move only to station 1 and hold there.
        tire_stop_2: move only to station 2 and hold there.
        after_tire_service: run stations 3 -> 4 -> 5 and finish.
        """
        route = str(route).strip()
        route_ranges = {
            'tire_stop_1': (0, 1),
            'tire_stop_2': (1, 2),
            'after_tire_service': (2, len(self.stations)),
        }
        if route not in route_ranges:
            return {
                'success': False,
                'code': 'UNKNOWN_ROUTE',
                'message': f'unsupported route: {route}',
            }
        if self.route_state == 'RUNNING':
            return {
                'success': False,
                'code': 'BUSY',
                'message': 'route is already running',
            }
        if self.route_state in ('STOPPED', 'ERROR'):
            return {
                'success': False,
                'code': 'RESET_REQUIRED',
                'message': f'RESET is required from {self.route_state}',
            }

        start_index, end_index = route_ranges[route]
        self._cancel_retry_timer()
        self._goal_seq += 1
        self.station_index = start_index
        self.route_end_index = end_index
        self.docking_sm = None
        self.wait_start_time = None
        self.current_goal_handle = None
        self.docking_triggered_early = False
        self._nav2_fail_count = 0
        self.route_name = route
        self.route_error = ''
        self.route_state = 'RUNNING'
        self.mode = 'NAV2'
        self.send_next_nav2_goal()
        return {'success': True, 'message': f'route started: {route}'}

    def stop_route(self):
        """현재 경로를 취소하고 늦은 callback이 주행을 재개하지 못하게 한다."""
        self._cancel_retry_timer()
        self._goal_seq += 1
        goal_handle = self.current_goal_handle
        self.current_goal_handle = None
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception as error:
                self.get_logger().error(f'Nav2 goal cancel 요청 실패: {error}')
        self.docking_sm = None
        self.wait_start_time = None
        self.docking_triggered_early = False
        self.mode = 'STOPPED'
        self.route_state = 'STOPPED'
        self.cmd_pub.publish(Twist())
        self.get_logger().warn('STOP: Nav2/도킹/재시도 중단, zero Twist 발행')
        return {'success': True, 'message': 'route stopped'}

    def reset_route(self):
        """STOPPED/ERROR를 IDLE로 되돌리되 경로는 시작하지 않는다."""
        if self.route_state not in ('STOPPED', 'ERROR'):
            return {
                'success': False,
                'code': 'INVALID_STATE',
                'message': f'RESET is not allowed from {self.route_state}',
            }
        self._cancel_retry_timer()
        self._goal_seq += 1
        self.station_index = 0
        self.docking_sm = None
        self.wait_start_time = None
        self.current_goal_handle = None
        self.docking_triggered_early = False
        self._nav2_fail_count = 0
        self.route_name = None
        self.route_error = ''
        self.mode = 'IDLE'
        self.route_state = 'IDLE'
        self.cmd_pub.publish(Twist())
        return {'success': True, 'message': 'route reset to IDLE'}

    def get_route_status(self):
        current_station = None
        if 0 <= self.station_index < len(self.stations):
            if self.route_state in ('RUNNING', 'STOPPED', 'ERROR'):
                current_station = self.stations[self.station_index].name
        return {
            'state': self.route_state,
            'current_station': current_station,
            'station_index': self.station_index,
            'total_stations': len(self.stations),
            'error': self.route_error or None,
            'route': self.route_name,
            'internal_state': self.mode,
        }

    def _set_route_error(self, message):
        self._cancel_retry_timer()
        self._goal_seq += 1
        self.current_goal_handle = None
        self.docking_sm = None
        self.wait_start_time = None
        self.route_error = str(message)
        self.route_state = 'ERROR'
        self.mode = 'ERROR'
        self.cmd_pub.publish(Twist())

    def _cancel_retry_timer(self):
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None

    def _schedule_retry(self, delay_sec, callback):
        if self.route_state != 'RUNNING':
            return
        self._cancel_retry_timer()
        self._retry_timer = self.create_timer(delay_sec, lambda: self._fire_once(callback))

    def _fire_once(self, callback):
        self._cancel_retry_timer()
        if self.route_state == 'RUNNING':
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
        """새 스테이션에 대한 목표 전송 시작점. 실패 카운터를 리셋한 뒤
        실제 전송은 _send_nav2_goal_internal()에 위임한다."""
        if self.route_state != 'RUNNING':
            return
        self._nav2_fail_count = 0
        self._send_nav2_goal_internal()

    def _send_nav2_goal_internal(self):
        if self.route_state != 'RUNNING':
            return
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
        self._goal_seq += 1
        my_seq = self._goal_seq

        if not self.nav2_client.server_is_ready():
            self.get_logger().error(
                f'Nav2 액션 서버 응답 없음, {GOAL_RETRY_DELAY_SEC}초 후 재시도')
            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self._retry_same_goal)
            return

        send_future = self.nav2_client.send_goal_async(goal_msg)
        send_future.add_done_callback(
            lambda future, seq=my_seq: self.nav2_goal_response_callback(future, seq))

    def _retry_same_goal(self):
        """실패한 목표를 실패 카운터는 유지한 채로 재전송.
        send_next_nav2_goal()은 카운터를 리셋하므로 재시도 경로에서는 쓰지 않는다."""
        self._cancel_retry_timer()
        self._send_nav2_goal_internal()

    def nav2_goal_response_callback(self, future, seq):
        if seq != self._goal_seq or self.route_state != 'RUNNING':
            try:
                stale_goal = future.result()
                if stale_goal.accepted:
                    stale_goal.cancel_goal_async()
            except Exception:
                pass
            return
        try:
            goal_handle = future.result()
        except Exception as error:
            self._set_route_error(f'Nav2 goal request failed: {error}')
            return
        if not goal_handle.accepted:
            self.get_logger().warn(
                f'Nav2 목표 거부됨, {GOAL_RETRY_DELAY_SEC}초 후 재시도')
            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self._retry_same_goal)
            return
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future, seq=seq: self.nav2_result_callback(future, seq))

    def nav2_result_callback(self, future, seq):
        if seq != self._goal_seq or self.route_state != 'RUNNING':
            return
        if self.docking_triggered_early:
            return
        # 이미 capture radius 진입으로 도킹으로 조기 전환된 경우, 뒤늦게 도착하는
        # (취소됐거나 정상 완료된) Nav2 결과 콜백은 무시한다. 안 그러면 이미 DOCKING
        # 모드로 넘어간 뒤에 advance_to_next_station이 다시 불려서 스테이션이 씹힐 수 있음.

        station = self.stations[self.station_index]
        result = future.result()

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self._nav2_fail_count += 1
            self.get_logger().error(
                f'[{station.name}] Nav2 목표 실패 (status={result.status}), '
                f'{self._nav2_fail_count}/{NAV2_MAX_RETRIES}회')

            if self._nav2_fail_count >= NAV2_MAX_RETRIES:
                self.get_logger().error(
                    f'[{station.name}] Nav2 목표 {NAV2_MAX_RETRIES}회 연속 실패, '
                    f'재시도 포기하고 정지')
                self._set_route_error(
                    f'{station.name}: Nav2 failed {NAV2_MAX_RETRIES} times'
                )
                return

            self._schedule_retry(GOAL_RETRY_DELAY_SEC, self._retry_same_goal)
            return

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
        if self.route_state != 'RUNNING':
            return
        self.docking_triggered_early = True
        self._goal_seq += 1

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
        if self.route_state != 'RUNNING':
            return
        self.station_index += 1
        if self.station_index >= self.route_end_index:
            if self.loop_mode and self.route_name == 'after_tire_service' and self.route_end_index >= len(self.stations):
                self.get_logger().info('한 바퀴 완료 -> 루프 재시작')
                self.station_index = 0
            else:
                self.get_logger().info('전체 경로 완료')
                self.mode = 'DONE'
                self.route_state = 'COMPLETED'
                self.current_goal_handle = None
                self.docking_sm = None
                self.cmd_pub.publish(Twist())
                return
        self.send_next_nav2_goal()

    def control_loop(self):
        if self.route_state in ('IDLE', 'COMPLETED', 'STOPPED', 'ERROR'):
            if self.route_state in ('STOPPED', 'ERROR'):
                self.cmd_pub.publish(Twist())
            return
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

        elif self.mode == 'ERROR':
            # Nav2 목표가 NAV2_MAX_RETRIES회 연속 실패해서 재시도 포기하고 정지한 상태.
            # 계속 정지 명령만 유지, 별도 자동 복구 없음 (사람 개입 필요).
            self.cmd_pub.publish(Twist())
            return


def main():
    rclpy.init()
    node = WaypointManager()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()