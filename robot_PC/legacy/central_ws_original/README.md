# roscamp-repo-2 — central_ws

ROS 2와 AI를 활용한 자율주행 로봇개발자 부트캠프 2팀의 무인정비소 중앙 제어 워크스페이스입니다.

Main PC ROS 2 workspace for central control.

## Packages

- `central_control`: parking-gated manual and automatic central workflows.
- `robot_interfaces`: shared interface package skeleton.

## External Endpoints

- Vision PC: `192.168.0.81:6000`, newline-delimited JSON.
- Left arm default: `192.168.0.115:5000`.
- Right arm default: `192.168.0.112:5001`.

Endpoint values live in `src/central_control/config/network.yaml` and can be overridden from launch arguments.

## Manual Flow

The existing manual test entry point is preserved:

```bash
ros2 launch central_control central_manual.launch.py
```

Manual flow still waits for operator input after parking completion.

## Auto Flow

The automatic operation node is separate:

`IDLE -> ARMED -> WAITING_FOR_VEHICLE -> PARKING_STABLE -> SETTING_CAMERA -> DETECTING_TIRES -> CHECKING_LEFT_TIRE -> EXECUTING_RIGHT_HELP -> EXECUTING_LEFT_REMOVE -> REQUESTING_PINKY_MOVE -> WAITING_PINKY_COMPLETE -> COMPLETED`

If `left.class_name == good_tire`, the node skips HELP/REMOVE and completes. If any detection result is `none`, unstable, malformed, or missing, it enters `ERROR`.

## Auto Keys

- `A`: arm/disarm automatic system.
- `N`: bypass parking while armed for test flow.
- `S`: priority STOP to both arms.
- `R`: reset after `ERROR`, `COMPLETED`, or `STOPPED`.
- `H`: manual HOME to both arms.
- `Q`/Esc: quit.

## Safe Mock Run

This does not move real arms:

```bash
cd /home/soo/robot_PC/central_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch central_control auto_operation.launch.py \
  mock_camera:=true \
  mock_vehicle_detector:=true \
  mock_arms:=true \
  mock_vision:=false \
  auto_start:=false \
  vision_host:=192.168.0.81 \
  vision_port:=6000
```

## Real Run

Use only after camera ROI, vision PC, and arm TCP servers are verified:

```bash
ros2 launch central_control auto_operation.launch.py \
  mock_camera:=false \
  mock_vehicle_detector:=false \
  mock_arms:=false \
  mock_vision:=false \
  auto_start:=false \
  vision_host:=192.168.0.81 \
  vision_port:=6000
```

## Checks

```bash
python3 -m compileall src/central_control
pytest -q
colcon build --symlink-install
source install/setup.bash
colcon test
colcon test-result --verbose
ros2 pkg executables central_control
nc -vz -w 3 192.168.0.81 6000
```

## Safety Notes

- `mock_arms: true` by default; real arms move only with `mock_arms:=false`.
- `auto_start: false` by default; operator must arm with `A` unless explicitly overridden.
- Parking completion alone never starts arm motion unless the auto system is armed.
- ERROR has no automatic retry; reset with `R`.

## Pinky route integration

After `REMOVE_BAD_TIRE` succeeds, `auto_operation` waits until `/pinky/route_command` has a subscriber, publishes `START:after_tire_service` exactly once, and waits for `/pinky/route_status`. `COMPLETED` completes the central cycle and `ERROR:<message>` enters `ERROR`. `timeouts.pinky_move_sec` in `config/auto_operation.yaml` defaults to 180 seconds. Manual STOP and vehicle-motion emergency both publish `STOP` to Pinky.

```bash
# Topic-only integration test
ros2 topic echo /pinky/route_command std_msgs/msg/String
ros2 topic pub --once /pinky/route_status std_msgs/msg/String "{data: RUNNING}"
ros2 topic pub --once /pinky/route_status std_msgs/msg/String "{data: COMPLETED}"
```


## Direct keyboard route test

```bash
cd /home/soo/robot_PC/central_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run central_control pinky_keyboard_trigger
```

`Space` sends `START:after_tire_service`, `s` sends `STOP`, and `q` exits. The trigger waits for a `/pinky/route_command` subscriber before START and suppresses duplicate START commands. Do not run it alongside `auto_operation` during an active automatic cycle.
