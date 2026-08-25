# central_control

`central_control` is the Main PC ROS 2 package.

Existing files kept for manual verification:

- `manual_central_node.py`
- `launch/central_manual.launch.py`

Automatic operation additions:

- `operation_controller.py`: auto operation state and safety transitions.
- `auto_operation_node.py`: ROS/OpenCV runtime for automatic cycle execution.
- `launch/auto_operation.launch.py`: final automatic launch file.
- `config/auto_operation.yaml`: automatic operation flags, delays, and command names.

Shared modules reused by both flows:

- `parking_monitor.py`
- `stop_detector.py`
- `vehicle_detector.py`
- `arm_client.py`
- `dual_arm_manager.py`
- `vision_client.py`

Parking ROI, top-camera calibration paths, BEV source points, and stop
thresholds remain in `config/parking.yaml`; vehicle model/mock settings remain
in `config/yolo_vehicle.yaml`.

Both `manual_central` and `auto_operation` process the top-camera image in the
same order: lens-undistort, perspective warp to 600 x 400 BEV, vehicle
detection, candidate ROI filtering, and full-entry ROI judgment. Blue is the
candidate-selection ROI and yellow is the full-entry ROI. The default
calibration files are `~/bev_map_calibration/cameraMatrix.npy` and
`~/bev_map_calibration/distCoeffs.npy`.

The launch files run the nodes with
the configured Python executable so real YOLO mode can import
Ultralytics. Override the `python_executable` launch argument if the virtual
environment moves. Direct `ros2 run` still uses the interpreter recorded when
the package was built, so use the launch files for real YOLO operation.

## Vehicle-motion safety interlock

After parking is confirmed, `auto_operation` stores the parking monitor's
filtered vehicle center as the safety reference. Monitoring is active only in
`SETTING_CAMERA`, `DETECTING_TIRES`, `CHECKING_LEFT_TIRE`,
`EXECUTING_RIGHT_HELP`, and `EXECUTING_LEFT_REMOVE`.

The safety tracker applies the same median filtering used by parking detection.
An emergency is latched only when jitter, regression speed, or displacement
from the parking reference remains above its configured threshold for
`emergency_hold_sec`. A missing vehicle detection is tolerated until
`detection_lost_emergency_sec`; a single failed camera frame or short YOLO miss
does not trigger the interlock.

A vehicle-motion emergency uses a dedicated executor and follows this sequence:

1. Latch the emergency and ignore all later normal-worker results.
2. Send `STOP` to both arms in parallel.
3. Wait `stop_to_home_delay_sec`.
4. Send `HOME` to both arms in parallel even if either `STOP` failed.
5. Remain in `EMERGENCY` until the operator presses `R`.

`S` remains the normal manual STOP and never starts HOME. While
`EMERGENCY_STOPPING` or `EMERGENCY_HOMING` is running, reset is rejected. After
final `EMERGENCY`, press `R`, confirm the vehicle is stationary, then press `A`
to arm a new cycle. The vehicle must satisfy parking detection again.

Safety defaults are in `config/auto_operation.yaml` and every safety setting is
also exposed as an `auto_operation.launch.py` argument:

```yaml
safety:
  vehicle_motion_monitor_enabled: true
  emergency_jitter_px: 12.0
  emergency_speed_px_sec: 6.0
  emergency_reference_displacement_px: 12.0
  emergency_hold_sec: 0.35
  detection_lost_emergency_sec: 1.0
  send_stop_before_home: true
  stop_to_home_delay_sec: 0.5
  home_on_vehicle_motion: true
  require_manual_reset: true
```

The OpenCV status window displays whether monitoring is active, reference and
current motion metrics, violation/lost timers, emergency reason, and per-arm
STOP/HOME results. `manual_central` does not use this automatic HOME interlock.


## Pinky route handoff

Topics:

- Publish `/pinky/route_command` (`std_msgs/msg/String`): `START:after_tire_service`, `STOP`.
- Subscribe `/pinky/route_status` (`std_msgs/msg/String`): `IDLE`, `RUNNING`, `COMPLETED`, `ERROR:<message>`.

Set `mock_transport_operations:=true` for full automatic operation without Pinky. In that mode `WAITING_TRANSPORT` is entered and immediately accepted without keyboard or Pinky input; the left/right ArUco PLACE/PICK services, arm commands, tire vision, parking detection, HOME, STOP, and emergency paths remain real. Set it to `false` to retain the external transport-ready/Pinky path.

Tire detection is accepted atomically: both `left` and `right` must have final
GOOD/BAD statuses. Any RECHECK, ERROR, or temporary miss discards both results
and requests a full cycle reset before both cameras are read again. The defaults
are `operation.detection_max_retries: 3` and
`operation.detection_retry_delay_sec: 0.5`. Exhaustion enters
`DETECTION_ERROR`; no HELP or REMOVE command is issued from that state.

Build and run:

```bash
cd robot_PC/central_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch central_control auto_operation.launch.py
```


## Pinky keyboard trigger

For a direct route-topic test that does not advance the automatic operation state machine, run the interactive trigger in its own terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run central_control pinky_keyboard_trigger
```

Keys:

- `Space`: publish `START:after_tire_service` once, only when a Pinky subscriber is connected.
- `s`: publish `STOP`.
- `q`: restore the terminal and quit.

The node also displays `/pinky/route_status`. Do not run `pinky_keyboard_trigger` and `auto_operation` as competing command sources during a real cycle. Running this node alone does not move Pinky; pressing Space can start physical motion.


## Pinky TCP bridge

auto_operation.launch.py는 pinky_tcp_bridge를 함께 실행합니다. bridge는 기존
/pinky/route_command (START:after_tire_service, STOP, RESET)를 network.yaml의
Pinky TCP endpoint로 변환하고, 원격 상태를 /pinky/route_status에 publish합니다.

단독 점검:

~~~bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run central_control pinky_tcp_bridge --ros-args \
  -p network_config:=ROBOT_PC_ROOT/central_ws/src/central_control/config/network.yaml
~~~

그 다음 기존 ros2 run central_control pinky_keyboard_trigger를 사용할 수 있습니다.
Space는 실제 START_ROUTE로 연결되므로 현장 안전 확인 전에는 누르지 마십시오.
