# Investigation Report

Date: 2026-07-20

## Source Workspace Surveyed

Read-only source: `/home/soo/tire_ws/src`

Packages found:

- `decision`: YOLO tire/car detection, parking judgment, manual test flows, TCP clients, remote tire detection server/client.
- `pick_aru`: ArUco marker detection and right-arm tracking helper code used by the manual multi-camera test.

## Key Existing Files

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/detector.py`

Purpose: reusable YOLO wrapper.

Reusable classes:

- `TireDetector`: loads an Ultralytics YOLO model and returns `detections, annotated_frame`.
- `VehicleDetector`: extends `TireDetector` and filters detections by vehicle class names.

Reusable methods:

- `TireDetector.detect(frame)`
- `VehicleDetector.detect(frame)`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/tcp_client.py`

Purpose: TCP robot-arm command client.

Reusable class:

- `TcpCommandClient`

Reusable methods:

- `send_command(command)`: sends to one or all configured targets.
- `send_command_to(target_name, command)`: sends to one named target.
- `_normalize_targets(targets)`: accepts dict or tuple target formats.

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/remote_tire_client.py`

Purpose: central-PC client for a remote tire YOLO detector server.

Reusable class:

- `RemoteTireDetectClient`

Reusable method:

- `detect(side="left")`: sends `DETECT_LEFT` or `DETECT_RIGHT`, parses `result:confidence`.

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/parking_monitor.py`

Purpose: parking ROI and stop-state judgment.

Reusable classes:

- `ParkingState`: `WAITING`, `ENTERING`, `STOP_CHECK`, `PARKED`, `TRIGGER`.
- `ParkingMonitor`: tracks vehicle center movement inside one or more parking polygons.

Reusable methods:

- `contains(point)`
- `reset_motion()`
- `update(center, now=None)`
- `consume_parked()`
- `stopped_seconds(now=None)`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/manual_trigger_test_node.py`

Purpose: single-camera manual Space/Enter trigger test.

Relevant behavior:

- Runs tire YOLO continuously.
- Uses `cv2.waitKey(1)` and triggers on Space, Enter, or Return.
- Maps `bad_tire` to `REMOVE_BAD_TIRE`.
- Maps `good_tire` to `HELP`.
- Sends commands through `TcpCommandClient` in a background thread.

Reusable methods:

- `publish_text(publisher, text)`
- `select_detection(detections)`
- `detection_to_command(detection_result)`
- `send_command_worker(command)`
- `trigger_current_detection()`
- `process_frame()`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/manual_multi_camera_test_node.py`

Purpose: staged manual multi-camera workflow.

Relevant behavior:

- Starts with top-view BEV parking check.
- Space confirms parking and opens tire cameras.
- Sends simultaneous `SET_CAMERA` to both arms.
- Runs left/right tire YOLO.
- Later Space triggers tire command sequence from latest snapshot.
- Guards against post-confirmation vehicle movement and sends `HOME`.
- Optional ArUco tracking stage after bad-tire removal.

Reusable functions/classes:

- `fourcc_to_string(value)`
- `ManualMultiCameraTestNode`

Reusable methods:

- Parameter/config: `declare_parameters_from_config`, `get_ros_parameter_overrides`, `normalize_enabled_arms`, `resolve_model_path`, `build_camera_configs`
- Camera/frame: `open_cameras`, `close_cameras`, `read_frame`, `to_bev`, `process_frames`, `process_parking`, `process_tire`
- Detection selection: `select_vehicle`, `select_tire_detection`
- Manual flow: `handle_key`, `confirm_parking_step`, `run_set_camera_step`, `reset_manual_state`, `start_manual_motion`, `run_motion_sequence`
- TCP/arm control: `send_tcp_command`, `send_to_arm`, `send_to_arm_worker`, `run_emergency_home`, `run_left_bad_tire_sequence_sync`
- Safety/ArUco: `handle_vehicle_motion_guard`, `start_aruco_tracking`, `process_aruco_frame`, `send_aruco_tracking_command`, `stop_right_arm_for_aruco`, `finish_aruco_tracking`, `fail_aruco_tracking`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/tire_detect_server.py`

Purpose: vision-side TCP server for tire detection.

Relevant behavior:

- Listens on configured host/port.
- Accepts `START_TIRE_DETECT`, `DETECT_LEFT`, and `DETECT_RIGHT`.
- Opens configured camera per side.
- Runs YOLO until a stable non-`none` result is found or timeout occurs.
- Returns `result:confidence` or `ERR:...`.

Reusable class:

- `TireDetectServer`

Reusable methods:

- `resolve_model_path(model_path, package_share)`
- `open_cameras(camera_indices=None)`
- `select_detection(detections)`
- `detect_once(side="left")`
- `combine_results(frame_results)`
- `serve_forever()`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/realtime_auto_node.py`

Purpose: automatic parking-gated tire workflow.

Reusable methods identified:

- `resolve_model_path`
- `publish_text`
- `select_detection`
- `detection_to_command`
- `update_stability`
- `can_send_command`
- `send_command_worker`
- `start_command`
- `remote_tire_detection_worker`
- `run_left_bad_tire_sequence`
- `start_remote_tire_detection`
- `to_bev`
- `select_vehicle`
- `send_set_camera_worker`
- `start_set_camera`
- `process_parking_frame`
- `start_tire_camera_mode`
- `process_frame`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/decision_logic.py`

Purpose: stable tire decision-to-command logic.

Reusable class:

- `TireDecisionLogic`

Reusable methods:

- `decide_raw(detections)`
- `decide_stable(detections)`

### `/home/soo/tire_ws/src/decision/tire_yolo_decision/set_camera_test_node.py`

Purpose: manual staged test that starts with `SET_CAMERA`, then executes tire-based motion.

Reusable methods:

- `publish_text`
- `select_detection`
- `detection_to_command`
- `send_command_worker`
- `start_set_camera`
- `start_detected_motion`
- `handle_trigger`
- `process_frame`

### `/home/soo/tire_ws/src/pick_aru/pick_aru/aruco_detector.py`

Purpose: ArUco marker detection helper.

Reusable class:

- `ArucoDetector`

Reusable methods:

- `detect(frame)`
- `_detect_once(...)`

### `/home/soo/tire_ws/src/pick_aru/pick_aru/right_arm_tracker.py`

Purpose: right-arm J1/J3 tracking helper for ArUco centering.

Reusable function/class:

- `clamp(value, min_value, max_value)`
- `RightArmArucoTracker`

## Suggested Split For Future Implementation

Central side:

- parking decision and workflow orchestration
- arm TCP command client
- remote tire detection client
- shared message/service definitions if needed

Vision side:

- YOLO model loading
- tire detection server
- camera capture and detection stabilization

No implementation was performed in this step.

## Vision Workspace Implementation

Date: 2026-07-20

Modified scope:

- `/home/soo/robot_PC/vision_ws`
- `/home/soo/robot_PC/REPORT.md`

Central workspace was not modified in this implementation step.

Implemented package:

- `tire_vision`: ROS 2 Python package for the vision-side tire detection service.

Implemented modules:

- `camera_manager.py`: fake camera, lazy OpenCV camera wrapper, camera manager.
- `tire_detector.py`: mock detector, lazy YOLO detector, detector factory.
- `detection_stabilizer.py`: repeated-result stabilization with bad-tire priority.
- `vision_service.py`: health and tire detection request handling with BUSY rejection.
- `vision_server_node.py`: ROS 2 node and newline-delimited JSON TCP server on port 6000.
- `health_check.py`: mock/model/uptime health status.
- `protocol.py`: request parsing and JSON-line response encoding.

Configuration and launch:

- `config/vision.yaml`: defaults to `mock_mode: true` and `tcp_port: 6000`.
- `launch/vision_server.launch.py`: launches `vision_server` with configurable `config_path`.

Tests:

- `test_protocol.py`
- `test_detection_stabilizer.py`
- `test_vision_service.py`
- `conftest.py`

Verification:

- `pytest -q /home/soo/robot_PC/vision_ws/src/tire_vision/test`
  - Result: `8 passed in 0.30s`
- `colcon build` from `/home/soo/robot_PC/vision_ws`
  - Result: `2 packages finished`

Notes:

- No real camera is opened at node startup.
- No YOLO model is loaded at node startup.
- The service can run without model weights when `mock_mode: true`.
- The existing empty `vision_detection` package remains present and builds alongside `tire_vision`.

## Central Workspace Full Manual Workflow Extension

Date: 2026-07-21

Scope modified:

- `/home/soo/robot_PC/central_ws`

Scope not modified:

- `/home/soo/tire_ws`
- `/home/soo/robot_PC/vision_ws`

Backup:

- Initial central package backup preserved at `/home/soo/robot_PC/backups/central_control.backup_20260721_parking_migration`.

Existing parking logic:

- Parking ROI and stop/motion thresholds in `config/parking.yaml` were not changed in this step.
- Existing `manual_central_node.py` parking judgment structure was kept and only connected to the new work state machine.
- The PARKED event remains latched and transitions only once until reset or detection-loss reset behavior.

New files:

- `central_control/vision_client.py`
- `central_control/arm_client.py`
- `central_control/dual_arm_manager.py`
- `central_control/work_state_machine.py`
- `central_control/key_handler.py`
- `config/network.yaml`
- `config/operation.yaml`
- `test/fake_vision_server.py`
- `test/fake_arm_server.py`
- `test/test_vision_client.py`
- `test/test_arm_client.py`
- `test/test_dual_arm_manager.py`
- `test/test_work_state_machine.py`
- `test/test_manual_central_runtime.py`
- `test/test_parking_monitor.py`
- `test/conftest.py`

Modified files:

- `central_control/manual_central_node.py`
- `launch/central_manual.launch.py`
- `setup.py`
- `package.xml`
- `/home/soo/robot_PC/central_ws/README.md`
- `/home/soo/robot_PC/central_ws/src/central_control/README.md`
- `/home/soo/robot_PC/REPORT.md`

Setup.py result:

- `package_name = "central_control"`
- `find_packages(exclude=["test"])`
- installs `package.xml`, `resource/central_control`, `config/*.yaml`, `launch/*.py`, and optional `models/**/*`
- console script: `manual_central = central_control.manual_central_node:main`

Launch arguments:

- `parking_config`
- `yolo_config`
- `network_config`
- `operation_config`
- `mock_camera`
- `mock_vehicle_detector`
- `mock_arms`
- `mock_vision`
- `vision_host`
- `vision_port`
- `left_arm_host`
- `left_arm_port`
- `right_arm_host`
- `right_arm_port`

Verification results:

- `python3 -m compileall src/central_control`: passed
- `pytest -q`: `38 passed in 6.20s`
- `rm -rf build install log`: completed for `central_ws` build artifacts
- `colcon build --symlink-install`: `2 packages finished` (final rerun: central_control 1.15s, robot_interfaces 0.22s)
- `colcon test`: `2 packages finished` (final rerun: central_control 6.58s, robot_interfaces 0.14s)
- `colcon test-result --verbose`: `Summary: 38 tests, 0 errors, 0 failures, 0 skipped`
- `ros2 pkg list | grep central_control`: `central_control`
- `ros2 pkg executables central_control`: `central_control manual_central`
- launch install check with `find -L`: `install/central_control/share/central_control/launch/central_manual.launch.py`
- config install check with `find -L`: `network.yaml`, `operation.yaml`, `parking.yaml`, `yolo_vehicle.yaml`

Vision PC checks:

- `nc -vz -w 3 192.168.0.81 6000`: failed, `Connection refused`
- health request to `192.168.0.81:6000`: no JSON response because the TCP connection was refused
- detect_tires request to `192.168.0.81:6000`: no JSON response because the TCP connection was refused

Mock full state flow result:

- `N -> SPACE -> SPACE -> SPACE -> SPACE -> SPACE`
- `WAIT_SET_CAMERA_CONFIRM -> WAIT_DETECTION_CONFIRM -> WAIT_ACTION_CONFIRM -> WAIT_ARUCO_CONFIRM -> WAIT_HOME_CONFIRM -> COMPLETED`
- mock left detection: `bad_tire`, confidence `0.91`, stable `true`
- mock right detection: `good_tire`, confidence `0.88`, stable `true`

Still requiring user/hardware verification:

- Real Vision PC server availability at `192.168.0.81:6000`.
- Actual health and detect_tires JSON response from the Vision PC.
- Real `/dev/video2` camera framing and ROI alignment.
- Real YOLO model path and class output against a vehicle.
- Left arm TCP server at the configured host/port.
- Right arm TCP server at the configured host/port.
- Real robot-arm motion with `mock_arms=false`.

## Central Auto Operation Node Addition

Date: 2026-07-22

Scope modified:

- `/home/soo/robot_PC/central_ws`

Scope preserved:

- Existing `manual_central_node.py` remains present and callable.
- Existing `central_manual.launch.py` remains present.
- `/home/soo/robot_PC/vision_ws` was not modified.
- `/home/soo/tire_ws` was not modified.

New files:

- `central_control/operation_controller.py`
- `central_control/auto_operation_node.py`
- `launch/auto_operation.launch.py`
- `config/auto_operation.yaml`
- `test/test_operation_controller.py`
- `test/test_auto_operation_runtime.py`

Modified files:

- `setup.py`
- `README.md`
- `/home/soo/robot_PC/central_ws/README.md`
- `/home/soo/robot_PC/REPORT.md`

Existing parking logic changed:

- No. The automatic node reuses `CentralParkingStateMachine`, `ParkingMonitor`, `StopDetector`, and `VehicleDetector` behavior already present in the central package. Parking ROI and stop/motion values in `parking.yaml` were not changed.

Automatic state flow:

- `IDLE`
- `ARMED`
- `WAITING_FOR_VEHICLE`
- `PARKING_STABLE`
- `SETTING_CAMERA`
- `DETECTING_TIRES`
- `CHECKING_LEFT_TIRE`
- `EXECUTING_RIGHT_HELP`
- `EXECUTING_LEFT_REMOVE`
- `COMPLETED`
- terminal safety states: `ERROR`, `STOPPED`

Safety devices implemented:

- `mock_arms: true` default.
- `auto_start: false` default.
- Arm motion starts only after the system is armed and parking is stable.
- Parking completion alone does not move arms.
- Cycle latch prevents duplicate execution for the same parked event.
- `S` sends priority STOP from a daemon thread even while a worker is active.
- Worker thread performs TCP/detection actions; ROS timer/OpenCV loop does not block on TCP.
- SET_CAMERA requires both arms to succeed.
- Vision errors, timeout, invalid JSON, request_id mismatch, missing left/right, `stable=false`, or `none` enter `ERROR`.
- The auto action policy uses left tire result only: left bad runs right HELP then left REMOVE; left good completes without HELP/REMOVE.
- Right HELP failure blocks left REMOVE.
- Left REMOVE failure enters `ERROR`.
- `ERROR` has no auto retry; reset requires `R`.

Launch arguments in `auto_operation.launch.py`:

- `parking_config`
- `yolo_config`
- `network_config`
- `operation_config`
- `auto_config`
- `mock_camera`
- `mock_vehicle_detector`
- `mock_arms`
- `mock_vision`
- `auto_start`
- `vision_host`
- `vision_port`
- `left_arm_host`
- `left_arm_port`
- `right_arm_host`
- `right_arm_port`

Setup.py result:

- Preserved: `manual_central = central_control.manual_central_node:main`
- Added: `auto_operation = central_control.auto_operation_node:main`
- Existing config/launch install rules install `auto_operation.yaml` and `auto_operation.launch.py`.

Verification results:

- `python3 -m compileall src/central_control`: passed
- `pytest -q`: `67 passed in 6.45s`
- `rm -rf build install log`: completed for `central_ws` build artifacts
- `colcon build --symlink-install`: `2 packages finished` (`central_control`, `robot_interfaces`)
- `colcon test`: `2 packages finished`
- `colcon test-result --verbose`: `Summary: 67 tests, 0 errors, 0 failures, 0 skipped`
- `ros2 pkg executables central_control`:
  - `central_control auto_operation`
  - `central_control manual_central`

Installed files checked:

- `install/central_control/share/central_control/launch/auto_operation.launch.py`
- `install/central_control/share/central_control/launch/central_manual.launch.py`
- `install/central_control/share/central_control/config/auto_operation.yaml`
- `install/central_control/share/central_control/config/network.yaml`
- `install/central_control/share/central_control/config/operation.yaml`
- `install/central_control/share/central_control/config/parking.yaml`
- `install/central_control/share/central_control/config/yolo_vehicle.yaml`

Vision PC check:

- `nc -vz -w 3 192.168.0.81 6000`: failed, `No route to host`
- health request: no JSON response because the host was unreachable
- detect_tires request: no JSON response because the host was unreachable

Mock auto integration result:

- Input sequence: `A`, `N`, then automatic worker progression.
- State flow: `WAITING_FOR_VEHICLE -> PARKING_STABLE -> SETTING_CAMERA -> DETECTING_TIRES -> EXECUTING_RIGHT_HELP -> EXECUTING_LEFT_REMOVE -> COMPLETED`
- Commands sent in mock run:
  - `left SET_CAMERA`
  - `right SET_CAMERA`
  - `right HELP`
  - `left REMOVE_BAD_TIRE`
- Mock left result: `bad_tire`, confidence `0.91`, stable `true`
- Mock right result: `good_tire`, confidence `0.88`, stable `true`

Actual hardware still not verified:

- Real Vision PC route and TCP server availability at `192.168.0.81:6000`.
- Real Vision PC health and detect_tires responses.
- Real top camera `/dev/video2` and ROI alignment.
- Real vehicle YOLO model execution/class output.
- Real left arm TCP command responses.
- Real right arm TCP command responses.
- Full cycle with `mock_arms=false`.
