## 실행 방법
0. SSH 접속 및 환경 설정
```bash
ssh pinky@192.168.4.1
source ~/pinky_pro/install/local_setup.bash
```
1. 로봇 하드웨어 구동
```bash
ros2 launch pinky_bringup bringup_robot.launch.xml
```
2. Nav2 내비게이션 스택 실행
```bash
ros2 launch pinky_navigation bringup_launch.xml \
    map:=/home/pinky/roscamp-repo-2/src/pinky_pro/pinky_goal_pid/map_4.yaml \
    use_sim_time:=False \
    use_composition:=True
```
3. 커스텀 내비게이션 노드 빌드 및 실행
```bash
cd ~/roscamp-repo-2
colcon build --packages-select pinky_goal_pid --symlink-install
source ~/roscamp-repo-2/install/local_setup.bash
ros2 run pinky_goal_pid nav2_waypt
```
4. 개발 중 빌드 명령어
```bash
# 코드 수정 후 재빌드 시 (--symlink-install: setup.py entry_points 변경 없으면 재빌드 불필요)
colcon build --packages-select pinky_goal_pid --symlink-install
colcon build --packages-select pinky_navigation --symlink-install
```

## 개발 히스토리
- v1: 고정 경로 PD 컨트롤러 (goal_pd.py)
- v2: PD + 대각 이동 + 루프 모드 (rpt_pd.py)
- v3 (현재): Nav2 + 커스텀 도킹 하이브리드 (nav2_waypt.py, dock_control.py)
  - 도킹 중 라이다 기반 장애물 정지-재개 로직 반영 (MOVE_FORWARD/BACKWARD/DIAGONAL 공통 적용)
  - crosstrack 보정 재설계 진행 중 (cross_nav2.py, 별도 파일로 분리 개발)
  - Nav2 목표 실패(status 미검증) 처리 로직 추가: 실패 시 최대 3회 재시도 후 ERROR 모드로 정지
  - 로봇 뒤쪽에 트렁크 장착으로 인한 footprint 변경 대응 (전장 확장, inflation_radius/footprint_padding 재조정)
  - 신규 고정 장애물 반영한 맵(map_4) 제작 및 적용

## 주요 설계 결정
- Nav2는 장애물 회피 접근(느슨한 tolerance)만 담당, 정밀 도킹은 별도 상태머신이 담당
- AMCL pose 업데이트 간격 문제(1~7초 지연)로 인한 도착 판정 실패 → capture_radius로 조기 핸드오프하여 해결
- use_composition:=True로 전환: controller_server 파라미터 타입 오타(min_theta_velocity_threshold)로 인한 컴포넌트 로드 실패가 원인이었음을 확인, 수정 후 정상 동작 확인됨 (기존 "composition이 CPU 93%+ 점유" 원인으로 지목했던 것과 실제로는 무관한 별개 이슈였음)
- 도킹 구간은 Nav2 관할 밖이라 라이다를 별도로 구독해, 현재 이동 방향 기준 콘 안의 최소거리로 정지-재개 판단
  - rplidar_link가 base_link 기준 180도 회전 마운트되어 있어 LIDAR_YAW_OFFSET(π) 보정 필요 (미보정 시 정면 장애물 미검출 확인됨)
- MOVE_FORWARD/BACKWARD는 goal_yaw 고정 후 진행축만 판정(단일축), MOVE_DIAGONAL은 매 tick atan2(dy,dx)로 목표 방향 재계산 — 비진행축 오차 처리 방식이 서로 다름
- **footprint / inflation_radius는 항상 함께 검토**: inflation_radius가 footprint로부터 계산되는 inscribed_radius보다 작으면 Nav2가 경고를 띄우며, 실제로 코스트맵 판정이 왜곡되어 장애물을 치워도 collision이 반복되는 현상으로 이어짐을 확인. footprint(특히 후방 길이)를 변경할 때는 inflation_radius도 반드시 그 이상으로 같이 맞출 것
- Nav2 result status 미검증 문제: nav2_result_callback이 goal 성공/실패를 확인하지 않고 콜백만 오면 도킹으로 진입하던 버그로, planning 실패 시에도 잘못된 위치를 기준으로 도킹이 시작되는 원인이었음. GoalStatus.STATUS_SUCCEEDED 검증 및 재시도 로직으로 수정

## 진행 중 이슈
- **crosstrack 보정 (cross_nav2.py)**: MOVE_FORWARD/BACKWARD의 비진행축(lateral) 오차 보정. 이전 두 차례 시도(정방향/반전 부호) 모두 발산하여 롤백, 기하학부터 재설계 중
- **AMCL 업데이트 지연**: update_min_a/d를 0.015로 낮춰 완화 시도했으나 여전히 저속 구간에서 판정 지연에 따른 오차 발생 확인됨 (로그 잔여오차와 실측 위치 불일치 사례) — 라이다 hz, DDS/QoS 등 추가 원인 조사 필요
- **좁은 통로 구간 주행**: return_to_start 인근 등 로봇 폭 대비 통로 여유가 매우 좁은 구간에서, footprint/inflation_radius를 정상 범위로 맞춘 상태에서도 여전히 회피 실패 가능성 있음 — 파라미터 튜닝으로 해결 가능한 범위인지, 좌표/경로 재설계가 필요한 물리적 한계인지 재확인 필요
- **스테이션별 capture_radius**: 이전 스테이션 도킹 종료 지점과 다음 스테이션 approach_pose가 매우 가까운 경우, Nav2의 제자리 회전(use_rotate_to_heading)이 좁은 공간에서 충돌 반복을 유발하는 문제 확인. Station 클래스에 스테이션별 capture_radius 필드를 추가해 해당 구간만 Nav2 회전 정렬을 건너뛰고 즉시 도킹 컨트롤러로 넘기는 방향으로 별도 파일에서 구현 예정 (아직 미반영)
- **IMU(BNO055) 통합**: `pinky_imu_bno055` 노드가 BNO055 리셋 폴링에서 행업 (SYS_STATUS=0x01, SYS_ERR=0x06, register map write error), I2C 100kHz로 낮춰도 미해결. `ekf.yaml`은 검증 전까지 gyro yaw rate만 사용하도록 보수적으로 구성

## TODO
- Station별 capture_radius 적용 (별도 파일에서 작업 예정)
- Jetcobot 작업 시작 신호 연동 (현재 고정 시간 대기, PC1 → Pinky 명령 트리거 미구현)
- Nav2 주행 중 장애물 회피 검증 (도킹 중 lidar 정지-재개는 완료, Nav2 구간은 별도 검증 필요)
- 좁은 통로 구간(return_to_start 인근) 경로/좌표 재설계 검토
- 로깅 레벨(log_level) info보다 낮춰서 부담 완화
- 불필요한 노드 정리
