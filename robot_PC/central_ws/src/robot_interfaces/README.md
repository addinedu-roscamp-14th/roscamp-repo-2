# robot_interfaces

Shared ROS 2 interfaces used by the central and arm-side packages.

`srv/TransportTask` carries a transport operation and slot and returns the
server-confirmed side, operation, slot, success, and diagnostic message.


The Pinky handoff uses standard `std_msgs/msg/String` topics and does not add a custom interface to this package.
