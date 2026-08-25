# Architecture Notes

## Intended Workspace Roles

`central_ws` is reserved for orchestration and robot command logic.

`vision_ws` is reserved for camera and model execution logic.

## Empty Packages Created

Central workspace:

- `central_control`: future central ROS 2 Python package.
- `robot_interfaces`: future shared message/service/action package.

Vision workspace:

- `vision_detection`: future vision ROS 2 Python package.

## Implementation Boundary

This step intentionally creates no functional nodes, launch behavior, TCP side effects, or YOLO execution code.
