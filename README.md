
## test용 (rpt_pd.py)

0
ssh pinky@192.168.4.1

1
source ~/pinky_pro/install/local_setup.bash
ros2 launch pinky_bringup bringup_robot.launch.xml

2
source ~/pinky_pro/install/local_setup.bash
ros2 launch pinky_navigation bringup_launch.xml     map:=/home/pinky/ddd.yaml     use_sim_time:=False

3
cd ~/pinky_pro
source ~/pinky_pro/install/local_setup.bash
ros2 run pinky_goal_pid rpt_pd




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

## 주요 설계 결정
- Nav2는 장애물 회피 접근(느슨한 tolerance)만 담당, 정밀 도킹은 별도 상태머신이 담당
- AMCL pose 업데이트 간격 문제(1~7초 지연)로 인한 도착 판정 실패 → capture_radius로 조기 핸드오프하여 해결
- use_composition:=False : Nav2 composition container가 라즈베리파이4에서 CPU 93%+ 점유 → 비활성화

## TODO
- Jetcobot 작업 시작 신호 연동 (현재 고정 시간 대기)
- 장애물 회피 검증 (SR_022, lidar 기반 정지)
