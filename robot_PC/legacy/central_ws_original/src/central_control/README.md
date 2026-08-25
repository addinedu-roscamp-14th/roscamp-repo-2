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
`/home/soo/yolo_tire_test/yolo_env/bin/python` so real YOLO mode can import
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

Automatic left-tire flow ends as `EXECUTING_LEFT_REMOVE -> REQUESTING_PINKY_MOVE -> WAITING_PINKY_COMPLETE -> COMPLETED`. The START command is emitted only after a subscriber connects and only once per cycle. Set `timeouts.pinky_move_sec` in `config/auto_operation.yaml` (default 180.0). Reset with `R` clears the Pinky command latch and timeout. This stage intentionally adds neither a dual-arm completion gate nor normal-cycle HOME.

Tire detection is accepted atomically: both `left` and `right` must have final
GOOD/BAD statuses. Any RECHECK, ERROR, or temporary miss discards both results
and requests a full cycle reset before both cameras are read again. The defaults
are `operation.detection_max_retries: 3` and
`operation.detection_retry_delay_sec: 0.5`. Exhaustion enters
`DETECTION_ERROR`; no HELP or REMOVE command is issued from that state.

Build and run:

```bash
cd /home/soo/robot_PC/central_ws
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
