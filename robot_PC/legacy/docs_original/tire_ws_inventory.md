# tire_ws Inventory

Source inspected: `/home/soo/tire_ws/src`

## Packages

- `decision`
  - ROS 2 Python package.
  - Contains YOLO detector wrappers, decision nodes, parking monitor, TCP clients, and tire detection server.
- `pick_aru`
  - ROS 2 Python package.
  - Contains ArUco detection and right-arm tracking helpers.

## Existing Manual Spacebar Flow

- `decision/tire_yolo_decision/manual_trigger_test_node.py`
  - Single camera.
  - `cv2.waitKey(1)` handles Space/Enter trigger and Q/ESC quit.
  - `trigger_current_detection()` sends the current tire result command.
- `decision/tire_yolo_decision/manual_multi_camera_test_node.py`
  - Multi-stage top camera plus tire cameras.
  - `handle_key()` handles Space, R, Q, ESC.
  - Space confirms parking first, then later triggers tire motion.
- `decision/tire_yolo_decision/set_camera_test_node.py`
  - Space/Enter starts staged `SET_CAMERA` and detected tire motion.

## Existing TCP Clients

- `decision/tire_yolo_decision/tcp_client.py`
  - `TcpCommandClient` sends robot-arm commands and waits for ACK.
- `decision/tire_yolo_decision/remote_tire_client.py`
  - `RemoteTireDetectClient` calls a remote tire detection TCP server.
- `manual_multi_camera_test_node.py`
  - Has additional direct `send_tcp_command()` and named left/right arm helpers.

## Existing Parking Judgment

- `decision/tire_yolo_decision/parking_monitor.py`
  - `ParkingMonitor` tracks whether the vehicle center remains stopped inside ROI long enough.
- `manual_multi_camera_test_node.py`
  - `process_parking()`, `select_vehicle()`, and `handle_vehicle_motion_guard()`.
- `realtime_auto_node.py`
  - `process_parking_frame()`, `select_vehicle()`, `start_tire_camera_mode()`.

## BEV and ROI Behavior Reused by `robot_PC`

- Source: `decision/tire_yolo_decision/realtime_auto_node.py` and
  `decision/config/yolo_config.yaml`.
- The top-camera frame is undistorted with `cameraMatrix.npy` and
  `distCoeffs.npy`, then warped from `bev_src_points` into a 600 x 400 BEV
  image before vehicle detection.
- `parking_detection_roi` filters vehicle candidates by bbox center.
- `parking_rois` checks full entry using the bbox front/rear points; these
  roles remain separate.
- The BEV source points, dimensions, and ROI coordinates are copied without
  recalibration so the central manual and automatic runtimes use the same
  pixel coordinate system as `tire_ws`.

## Existing YOLO Detection

- `decision/tire_yolo_decision/detector.py`
  - `TireDetector`
  - `VehicleDetector`
- `decision/tire_yolo_decision/tire_detect_server.py`
  - Vision-side TCP tire detector.
- Config and models:
  - `decision/config/yolo_config.yaml`
  - `decision/models/Car_decision/best.pt`
  - `decision/models/Tire_decision/best.pt`

## Do Not Modify

`/home/soo/tire_ws` must remain read-only during the next implementation stage unless explicitly instructed otherwise.
