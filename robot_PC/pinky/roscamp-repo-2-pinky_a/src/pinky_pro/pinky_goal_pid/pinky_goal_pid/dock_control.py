#!/usr/bin/env python3
"""
정밀 도킹 상태 머신.
goal_pid.py의 ROTATE/MOVE(단일축 + 대각선) 로직을 그대로 가져오되,
"영원히 도는 노드"가 아니라 "한 세트의 웨이포인트를 끝내면 done=True를 반환하는
재사용 가능한 클래스"로 감쌌다.

[장애물 정지-재개]
step()에 min_obstacle_dist를 넘기면 docking_obstacle_stop_dist보다 가까울 때
선속도만 0으로 죽인다. MOVE_FORWARD/MOVE_BACKWARD/MOVE_DIAGONAL 세 갈래 모두 적용.

[Crosstrack 보정 - 시도했으나 롤백됨, 재시도 금지 전까지는 넣지 말 것]

[도착/오버슈트 판정 시 즉시 정지로 변경]
기존엔 도착/오버슈트 판정된 틱에서도 _rate_limit()으로 감속을 거는 방식이라,
로그에 찍히는 "잔여오차"가 판정 시점 값일 뿐이고 그 이후 관성으로 밀리는
거리(실측 시 3~4cm 이상)가 전혀 로그에 반영되지 않는 문제가 있었음.
로그값과 실측 최종 위치가 크게 어긋나는 원인이었음.
이제 도착/오버슈트 판정 시 rate_limit 없이 즉시 target_lin=0, target_ang=0으로
반환 -> 급정지지만 로그의 "잔여오차"가 실제 최종 오차와 일치하게 됨.
"""
import math
from geometry_msgs.msg import Twist


class DockingStateMachine:
    """
    waypoints: goal_pid.py와 동일한 포맷
        (x, y, target_yaw[rad], move_type, [wait_seconds])
        move_type: 'MOVE_FORWARD' / 'MOVE_BACKWARD' / 'ROTATE' / 'MOVE_DIAGONAL' / 'WAIT'
    params: dict, goal_pid.py의 declare_parameter 목록과 동일한 키 +
        'max_lin_accel'(m/s^2), 'max_ang_accel'(rad/s^2) - 속도 변화율 제한값
        'docking_obstacle_stop_dist'(m) - 이 거리보다 가까우면 정지 (기본 0.15)
    """

    def __init__(self, waypoints, params, logger, clock):
        self.waypoints = waypoints
        self.p = params
        self.logger = logger
        self.clock = clock
        self.reset()

    def reset(self):
        self.index = 0
        self.state = 'ROTATE'
        self.prev_signed_error = None
        self.prev_distance = None
        self.prev_move_time = None
        self.diag_heading_vec = None
        self.wait_start_time = None
        self.done = False
        self._prev_tick_time = None
        self._obstacle_warned = False

    def _finish_waypoint(self):
        self.index += 1
        self.state = 'ROTATE'
        self.prev_signed_error = None
        self.prev_distance = None
        self.prev_move_time = None
        self.diag_heading_vec = None
        self._obstacle_warned = False

    def _rate_limit(self, target_lin, target_ang, current_lin_vel, current_ang_vel):
        """실측 속도(odom) 기준으로 max_lin_accel/max_ang_accel만큼만 변하도록 제한."""
        now = self.clock.now()
        if self._prev_tick_time is None:
            dt = 0.05
        else:
            dt = max((now - self._prev_tick_time).nanoseconds / 1e9, 0.001)
        self._prev_tick_time = now

        max_dlin = self.p.get('max_lin_accel', 0.3) * dt
        max_dang = self.p.get('max_ang_accel', 2.0) * dt

        lin_diff = target_lin - current_lin_vel
        lin_diff = max(-max_dlin, min(max_dlin, lin_diff))
        limited_lin = current_lin_vel + lin_diff

        ang_diff = target_ang - current_ang_vel
        ang_diff = max(-max_dang, min(max_dang, ang_diff))
        limited_ang = current_ang_vel + ang_diff

        return limited_lin, limited_ang

    def _immediate_stop_cmd(self):
        """도착/오버슈트 판정 시 rate_limit 없이 즉시 정지 명령 반환.
        로그에 찍히는 잔여오차가 실제 최종 위치와 일치하도록 보장하기 위함."""
        self._prev_tick_time = self.clock.now()  # 다음 웨이포인트 rate_limit dt 계산 기준점 갱신
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def get_move_direction_relative(self, current_x, current_y, current_yaw):
        """현재 MOVE 상태에서 이동 중인 방향을 로봇 기준 각도(rad, 0=전방)로 반환.
        ROTATE 상태거나 완료됐으면 None. 라이다 콘 방향 잡는 용도로 노드에서 호출."""
        if self.done or self.index >= len(self.waypoints):
            return None
        if self.state != 'MOVE':
            return None

        wp = self.waypoints[self.index]
        goal_x, goal_y, goal_yaw, move_type = wp[0], wp[1], wp[2], wp[3]

        if move_type == 'MOVE_FORWARD':
            return 0.0
        elif move_type == 'MOVE_BACKWARD':
            return math.pi
        elif move_type == 'MOVE_DIAGONAL':
            dx = goal_x - current_x
            dy = goal_y - current_y
            world_angle = math.atan2(dy, dx)
            return math.atan2(
                math.sin(world_angle - current_yaw), math.cos(world_angle - current_yaw))
        return None

    def step(self, current_x, current_y, current_yaw, current_lin_vel, current_ang_vel,
              min_obstacle_dist=None):
        """control loop 1틱 실행. (Twist, done:bool) 반환.
        min_obstacle_dist: 현재 이동 방향 전방 콘 안의 최소 라이다 거리(m). None이면 체크 안 함."""
        target_lin = 0.0
        target_ang = 0.0

        if self.done or self.index >= len(self.waypoints):
            self.done = True
            lin, ang = self._rate_limit(0.0, 0.0, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, True

        wp = self.waypoints[self.index]
        goal_x, goal_y, goal_yaw, move_type = wp[0], wp[1], wp[2], wp[3]
        wait_seconds = wp[4] if len(wp) > 4 else 0.0

        if move_type == 'WAIT':
            if self.wait_start_time is None:
                self.wait_start_time = self.clock.now()
                self.logger.info(f'도킹 웨이포인트 {self.index} WAIT 시작 ({wait_seconds:.1f}초)')
            elapsed = (self.clock.now() - self.wait_start_time).nanoseconds / 1e9
            if elapsed >= wait_seconds:
                self.index += 1
                self.state = 'ROTATE'
                self.wait_start_time = None
            lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, False

        is_move_waypoint = move_type in ('MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_DIAGONAL')

        if move_type == 'MOVE_DIAGONAL':
            dx0 = goal_x - current_x
            dy0 = goal_y - current_y
            effective_target_yaw = math.atan2(dy0, dx0)
        else:
            effective_target_yaw = goal_yaw

        if self.state == 'ROTATE':
            yaw_error = math.atan2(
                math.sin(effective_target_yaw - current_yaw),
                math.cos(effective_target_yaw - current_yaw))
            yaw_error_deg = math.degrees(abs(yaw_error))

            finished_rotate = False
            if is_move_waypoint:
                if yaw_error_deg < self.p['move_start_yaw_threshold_deg']:
                    self.logger.info(f'도킹 웨이포인트 {self.index} {move_type}, 헤딩 정렬 확인, 이동 시작')
                    self.state = 'MOVE'
                    self.prev_signed_error = None
                    self.prev_distance = None
                    self.prev_move_time = None
                    self._obstacle_warned = False
                    if move_type == 'MOVE_DIAGONAL':
                        dxs = goal_x - current_x
                        dys = goal_y - current_y
                        norm = math.hypot(dxs, dys)
                        self.diag_heading_vec = (1.0, 0.0) if norm < 1e-6 else (dxs / norm, dys / norm)
                    finished_rotate = True
            else:
                if abs(yaw_error) < self.p['yaw_tolerance']:
                    self.logger.info(f'도킹 웨이포인트 {self.index} 방향 정렬 완료')
                    self.index += 1
                    self.state = 'ROTATE'
                    finished_rotate = True

            if not finished_rotate:
                ang_z = self.p['kp_angle'] * yaw_error
                if abs(ang_z) < self.p['min_ang_vel']:
                    ang_z = self.p['min_ang_vel'] if ang_z > 0 else -self.p['min_ang_vel']
                target_ang = max(-self.p['max_ang'], min(self.p['max_ang'], ang_z))

            lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, False

        elif self.state == 'MOVE':
            obstacle_stop_dist = self.p.get('docking_obstacle_stop_dist', 0.15)
            blocked = (min_obstacle_dist is not None and min_obstacle_dist < obstacle_stop_dist)
            if blocked and not self._obstacle_warned:
                self.logger.warn(
                    f'도킹 웨이포인트 {self.index} 이동 방향 장애물 감지 '
                    f'({min_obstacle_dist:.2f}m < {obstacle_stop_dist:.2f}m), 정지')
                self._obstacle_warned = True
            elif not blocked and self._obstacle_warned:
                self.logger.info(f'도킹 웨이포인트 {self.index} 장애물 해소, 이동 재개')
                self._obstacle_warned = False

            dx = goal_x - current_x
            dy = goal_y - current_y

            if move_type == 'MOVE_DIAGONAL':
                distance = math.hypot(dx, dy)
                hx, hy = self.diag_heading_vec if self.diag_heading_vec else (1.0, 0.0)
                projection = dx * hx + dy * hy

                if distance < self.p['goal_tolerance'] or projection <= 0.0:
                    self.logger.info(f'도킹 웨이포인트 {self.index} 위치 도착! (잔여오차 {distance:.4f}m)')
                    self._finish_waypoint()
                    return self._immediate_stop_cmd(), False

                now = self.clock.now()
                dt = 0.05 if self.prev_move_time is None else max(
                    (now - self.prev_move_time).nanoseconds / 1e9, 0.05)
                self.prev_move_time = now
                dist_deriv = 0.0 if self.prev_distance is None else (distance - self.prev_distance) / dt
                self.prev_distance = distance

                live_heading = math.atan2(dy, dx)
                yaw_error = math.atan2(
                    math.sin(live_heading - current_yaw), math.cos(live_heading - current_yaw))
                ang_z = self.p['kp_angle_diagonal'] * yaw_error
                target_ang = max(-self.p['max_ang_diagonal'], min(self.p['max_ang_diagonal'], ang_z))

                if blocked:
                    target_lin = 0.0
                else:
                    raw_vel = self.p['kp_dist'] * distance + self.p['kd_dist'] * dist_deriv
                    target_lin = min(self.p['max_lin'], raw_vel)
                    if target_lin < self.p['min_lin_vel']:
                        target_lin = self.p['min_lin_vel']

                lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
                cmd = Twist()
                cmd.linear.x = lin
                cmd.angular.z = ang
                return cmd, False

            # MOVE_FORWARD / MOVE_BACKWARD (단일축)
            if abs(math.cos(goal_yaw)) >= abs(math.sin(goal_yaw)):
                signed_error = dx
            else:
                signed_error = dy
            distance = abs(signed_error)

            overshoot = False
            if self.prev_signed_error is not None and self.prev_signed_error != 0:
                if (self.prev_signed_error > 0) != (signed_error > 0):
                    overshoot = True

            if distance < self.p['goal_tolerance'] or overshoot:
                if overshoot:
                    self.logger.warn(
                        f'도킹 웨이포인트 {self.index} 목표 지나침 감지(오버슈트), '
                        f'잔여오차 {distance:.4f}m -> 도착 처리')
                else:
                    self.logger.info(f'도킹 웨이포인트 {self.index} 위치 도착! (잔여오차 {distance:.4f}m)')
                self._finish_waypoint()
                return self._immediate_stop_cmd(), False

            self.prev_signed_error = signed_error

            now = self.clock.now()
            dt = 0.05 if self.prev_move_time is None else max(
                (now - self.prev_move_time).nanoseconds / 1e9, 0.05)
            self.prev_move_time = now
            dist_deriv = 0.0 if self.prev_distance is None else (distance - self.prev_distance) / dt
            self.prev_distance = distance

            raw_vel = self.p['kp_dist'] * distance + self.p['kd_dist'] * dist_deriv
            raw_target_lin = min(self.p['max_lin'], raw_vel)
            if raw_target_lin < self.p['min_lin_vel']:
                raw_target_lin = self.p['min_lin_vel']
            target_lin = -raw_target_lin if move_type == 'MOVE_BACKWARD' else raw_target_lin

            yaw_error = math.atan2(math.sin(goal_yaw - current_yaw), math.cos(goal_yaw - current_yaw))
            yaw_error_deg = math.degrees(abs(yaw_error))
            if yaw_error_deg < self.p['move_yaw_deadzone_deg']:
                target_ang = 0.0
            else:
                correction = self.p['move_yaw_correction_gain'] * yaw_error
                target_ang = max(-self.p['move_yaw_correction_limit'],
                                  min(self.p['move_yaw_correction_limit'], correction))

            if blocked:
                target_lin = 0.0

            lin, ang = self._rate_limit(target_lin, target_ang, current_lin_vel, current_ang_vel)
            cmd = Twist()
            cmd.linear.x = lin
            cmd.angular.z = ang
            return cmd, False

        lin, ang = self._rate_limit(0.0, 0.0, current_lin_vel, current_ang_vel)
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd, False