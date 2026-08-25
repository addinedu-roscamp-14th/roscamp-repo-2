# 배포

모든 명령은 통합 저장소 루트에서 실행합니다. 배포 스크립트는 기본 dry-run입니다.
실제 반영할 때만 --execute를 사용합니다.

## 중앙제어 PC

배포: central_ws, docs

    scripts/deploy_central.sh operator@central:/opt/robot_PC
    scripts/deploy_central.sh --execute operator@central:/opt/robot_PC

빌드:

    cd /opt/robot_PC/central_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install

실행:

    source install/setup.bash
    ros2 launch central_control auto_operation.launch.py mock_camera:=false mock_vehicle_detector:=false mock_arms:=false mock_vision:=false auto_start:=true

## 타이어 검출 PC

배포: vision_ws, docs. YOLO 가중치는 vision_ws/src/tire_vision/models에 포함됩니다.

    scripts/deploy_vision.sh operator@vision:/opt/robot_PC
    scripts/deploy_vision.sh --execute operator@vision:/opt/robot_PC

빌드:

    cd /opt/robot_PC/vision_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install

실행:

    source install/setup.bash
    ros2 launch tire_vision vision_server.launch.py mock_mode:=false

기본 검출 카메라는 left index 2, right index 4입니다.

## 로봇팔 비전 PC

배포: arm_vision_ws, calibration, docs

    scripts/deploy_arm_vision.sh operator@arm-vision:/opt/robot_PC
    scripts/deploy_arm_vision.sh --execute operator@arm-vision:/opt/robot_PC

빌드:

    cd /opt/robot_PC/arm_vision_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install

실행 예:

    source install/setup.bash
    ros2 launch arm_aruco_perception dual_aruco.launch.py left_camera_device:=LEFT_DEVICE right_camera_device:=RIGHT_DEVICE left_camera_calibration_file:=/opt/robot_PC/calibration/left/camera_calibration.npz right_camera_calibration_file:=/opt/robot_PC/calibration/right/camera_calibration.npz

실제 장치명과 왼쪽 보정 파일이 없으면 의도적으로 시작 오류가 발생합니다.
arm_task_coordinator는 상위 작업 요청에서 ReplacementCoordinator를 생성해 사용하며,
좌우 TCP 설정은 설치 share의 config/network.yaml입니다.

## 왼쪽 Raspberry Pi

배포: raspberry_left_ws, docs. 포트 5000.

    scripts/deploy_left_arm.sh pi@left-arm:/opt/robot_PC
    scripts/deploy_left_arm.sh --execute pi@left-arm:/opt/robot_PC

빌드/실행:

    cd /opt/robot_PC/raspberry_left_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    source install/setup.bash
    ros2 launch tire_arm_control arm_server.launch.py

## 오른쪽 Raspberry Pi

배포: raspberry_right_ws, docs. 포트 5001.

    scripts/deploy_right_arm.sh pi@right-arm:/opt/robot_PC
    scripts/deploy_right_arm.sh --execute pi@right-arm:/opt/robot_PC

빌드/실행:

    cd /opt/robot_PC/raspberry_right_ws
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    source install/setup.bash
    ros2 launch tire_arm_control arm_server.launch.py

## 전송 제외

공통 스크립트는 build, install, log, __pycache__, .pytest_cache, *.pyc, .git,
legacy를 제외합니다. 원본 legacy는 개발 저장소에만 보존하고 장비로 배포하지 않습니다.
