# left_arm_transport_pick

운반차량의 ArUco 마커를 왼팔 그리퍼 카메라로 검출하고, 핸드아이와 슬롯 등록 포즈로 왼팔 타이어 운반 목표를 계산하는 독립 ROS 2 패키지다.

## 현재 테스트 PC

현재 프로젝트 절대경로는 `/home/choiminjoon/robot_PC`다. 테스트 보정 파일은 다음 위치에 복사되어 있다.

```text
config/calibration/left/camera_calibration.npz
config/calibration/left/handeye_calibration.npz
config/calibration/left/slot_registration.json
```

`handeye_calibration.npz`는 legacy right-arm의 `handeye_calibration_FINAL.npz`를 복사한 것이다. NPZ 키는 `camera_matrix`, `dist_coeffs`, `T_gripper_camera` 등을 실제 파일에서 확인했다.

## 구조와 재사용

`aruco_detector.py`는 기존 `arm_aruco_perception`의 검출기와 PoseStabilizer를 재사용한다. `coordinate_transform.py`는 기존 HandEyeTransform을 사용하고, `robot_client.py`는 기존 `arm_task_coordinator.tcp_arm_client.TcpArmClient`를 감싼다. 슬롯 포즈와 집기 순서는 `slot_registration.py`와 `pick_sequence.py`로 분리했다.

현재 파일시스템에는 요청된 `right_arm_final/right_arm_client.py`가 없었다. 대신 `legacy/right_arm_original/charuco2/aruco_tire_target_planner.py` 및 관련 최신 슬롯/핸드아이 데이터를 기준으로 통합했다.

기존 Raspberry Pi TCP 서버, 서비스 파일, motions 파일은 수정하지 않았다. `POSE`/`PING`은 `GET_COORDS`, `MOVE_SINGLE_COORD`는 `GET_COORDS` 후 한 축을 바꾼 `MOVE_COORDS`로 클라이언트에서 변환한다. 모든 JSON 요청은 version/request_id/command를 포함하며 기존 TcpArmClient처럼 요청마다 새 연결을 사용한다. 정상 종료에서는 `STOP`을 보내지 않는다.

## 빌드와 preview 실행

```bash
cd /home/choiminjoon/robot_PC/arm_vision_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch left_arm_transport_pick left_arm_transport_pick.launch.py \
  camera_device:=/dev/video2 robot_host:=192.168.0.63 robot_port:=5000 \
  preview_only:=true allow_robot_motion:=false allow_gripper_close:=false
```

preview는 카메라 검출·안정화·목표 좌표 계산까지만 수행하고 로봇 TCP를 호출하지 않는다. 서비스 `/left_arm/pick_transport_slot`은 `std_srvs/Trigger`이며 슬롯은 `slot_id` 파라미터로 지정한다.

실제 이동은 `preview_only:=false allow_robot_motion:=true allow_gripper_close:=true`를 사용자가 명시적으로 지정해야 한다. 좌표의 NaN/Inf, 범위 초과, 캘리브레이션/슬롯/마커 불안정은 모두 이동을 차단한다.

## 담당자 PC 전달 시

`config/left_arm_transport_pick.yaml`의 `camera_calibration_file`, `handeye_file`, `slot_registration_file` 세 키를 담당자 PC의 `/home/choiminjoon/...` 경로로 변경한다. launch는 설치된 YAML을 읽으므로 Python 코드나 launch 파일에는 머신별 절대경로를 추가하지 않는다.

## 검증 범위와 주의

현재 단계에서는 실제 네트워크·카메라·로봇을 연결하지 않는다. Python compileall, ROS 2 build, 캘리브레이션 키/JSON 로딩, 가짜 TCP 서버를 통한 명령 형식 검증만 수행한다. 실제 장비 시험 전에는 카메라 장치, TCP 주소, 좌표계(mm/deg), 접근/집기 높이, 슬롯별 포즈, emergency latch 상태를 담당자가 확인해야 한다. `STOP`은 비상 상황에서만 명시적으로 호출한다.
