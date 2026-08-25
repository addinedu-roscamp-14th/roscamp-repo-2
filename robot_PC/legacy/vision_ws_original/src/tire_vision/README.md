# tire_vision

Vision-side ROS 2 Python package for tire detection.

The TCP service listens on port `6000` by default and accepts newline-delimited JSON.

Supported requests:

```json
{"type":"health"}
{"type":"detect_tires","side":"left"}
```

`mock_mode: true` is the default, so the service can run without YOLO weights or a real camera.

## Tire marker standalone test

The production detector and `tire_marker_test` measure the radial gap between a
yellow marker and the tire edge. They use a YOLO11 Small segmentation model with
one class, `tire`. Within the upper part of the tire mask they detect the marker
using configurable HSV thresholds and morphology.

```bash
cd ~/robot_PC/vision_ws
source ~/venv/yolo/bin/activate
python3 src/tire_vision/tire_vision/tire_marker_test.py \
  --model /home/seohyun/robot_PC/vision_ws/models/tire_best.pt --source 2
```

After a ROS 2 build it can also be run as:

```bash
colcon build --packages-select tire_vision --symlink-install
source install/setup.bash
ros2 run tire_vision tire_marker_test \
  --model /home/seohyun/robot_PC/vision_ws/models/tire_best.pt --source 2
```

`--source` accepts a camera index, image, or video. Press `q` or Escape to
quit, `r` to clear history, and `s` to save the displayed frame under
`~/robot_PC/vision_ws/debug_frames/`. CSV rows are appended to
`~/robot_PC/vision_ws/tire_marker_test.csv` by default. `--save-video PATH`
saves annotated video and `--no-display` disables the GUI.

The measured defaults are `bad_max_ratio=0.23`,
`good_min_ratio=0.30`, and `max_ratio_std=0.01`. A median at or below the
BAD threshold is BAD, a median at or above the GOOD threshold is GOOD, and the
interval between them is RECHECK. Fewer than 10 valid frames or excessive
standard deviation is also RECHECK. These values can be changed with
`--bad-max-ratio`, `--good-min-ratio`, and `--max-ratio-std`.

Reusable processing is split between `marker_detector.py` and
`wear_measurement.py`. `YoloTireDetector` keeps independent 15-value ratio
histories for the left and right cameras. A stable BAD ratio is returned as
`bad_tire`, and a stable GOOD ratio is returned as `good_tire`. Missing tires,
missing masks or markers, excessive variation, and the RECHECK interval return
an empty detections list.

`vision_service`, `vision_server_node`, the TCP protocol, and their response
schema are unchanged. The existing mock detector remains available through
`mock_mode: true`.

Each `left` and `right` result now includes a detection `status`: `GOOD`, `BAD`,
`RECHECK`, or `ERROR`. `class_name` remains `good_tire` for GOOD,
`bad_tire` for BAD, and `none` otherwise. A `detect_tires` request with
`reset_cycle: true` clears both ratio histories and both label stabilizers before
camera indexes 2 and 4 are processed again.
