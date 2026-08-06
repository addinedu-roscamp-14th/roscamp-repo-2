
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
    map:=/home/pinky/map_2.yaml \
    use_sim_time:=False \
    use_composition:=False
```
3. 커스텀 내비게이션 노드 빌드 및 실행
```bash
cd ~/pinky_pro
colcon build --packages-select pinky_goal_pid --base-paths src
source ~/pinky_pro/install/local_setup.bash
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

## 주요 설계 결정
- Nav2는 장애물 회피 접근(느슨한 tolerance)만 담당, 정밀 도킹은 별도 상태머신이 담당
- AMCL pose 업데이트 간격 문제(1~7초 지연)로 인한 도착 판정 실패 → capture_radius로 조기 핸드오프하여 해결
- use_composition:=False : Nav2 composition container가 라즈베리파이4에서 CPU 93%+ 점유 → 비활성화
- 도킹 구간은 Nav2 관할 밖이라 라이다를 별도로 구독해, 현재 이동 방향 기준 콘 안의 최소거리로 정지-재개 판단
  - rplidar_link가 base_link 기준 180도 회전 마운트되어 있어 LIDAR_YAW_OFFSET(π) 보정 필요 (미보정 시 정면 장애물 미검출 확인됨)
- MOVE_FORWARD/BACKWARD는 goal_yaw 고정 후 진행축만 판정(단일축), MOVE_DIAGONAL은 매 tick atan2(dy,dx)로 목표 방향 재계산 — 비진행축 오차 처리 방식이 서로 다름

## 진행 중 이슈
- **crosstrack 보정 (cross_nav2.py)**: MOVE_FORWARD/BACKWARD의 비진행축(lateral) 오차 보정. 이전 두 차례 시도(정방향/반전 부호) 모두 발산하여 롤백, 기하학부터 재설계 중
- **AMCL 업데이트 지연**: 최대 4.9초까지 발생, use_composition:=True 및 max_particles 축소로도 해소 안 됨 — 라이다 hz, DDS/QoS 등 추가 원인 조사 필요
- **IMU(BNO055) 통합**: `pinky_imu_bno055` 노드가 BNO055 리셋 폴링에서 행업 (SYS_STATUS=0x01, SYS_ERR=0x06, register map write error), I2C 100kHz로 낮춰도 미해결. `ekf.yaml`은 검증 전까지 gyro yaw rate만 사용하도록 보수적으로 구성

## TODO
- Jetcobot 작업 시작 신호 연동 (현재 고정 시간 대기, PC1 → Pinky 명령 트리거 미구현)
- Nav2 주행 중 장애물 회피 검증 (도킹 중 lidar 정지-재개는 완료, Nav2 구간은 별도 검증 필요)
- 로깅 레벨(log_level) info보다 낮춰서 부담 완화
- 불필요한 노드 정리
