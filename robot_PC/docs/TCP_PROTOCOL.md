# TCP 프로토콜

모든 요청과 응답은 UTF-8 한 줄이며 LF로 끝납니다. JSON protocol version은 1입니다.

## 기존 plain-text arm 명령

HOME, HELP, REMOVE_BAD_TIRE, STOP을 계속 지원합니다. 기존 SET_CAMERA와 우측
SET_ARUCO motion도 호환용으로 읽을 수 있지만 중앙제어 workflow는 SET_CAMERA를
전송하지 않습니다.

성공은 OK:motion_name, 실패는 ERR:CODE 또는 ERR:CODE:detail 형식입니다.

## JSON arm 명령

공통 요청:

    {"version":1,"request_id":"id","command":"COMMAND", ...}

지원 명령:

- MOVE_COORDS: coords 6개, speed 1..100, mode 0 또는 1, 선택 timeout_sec
- GET_COORDS
- GET_ANGLES
- GRIPPER_OPEN
- GRIPPER_CLOSE
- PLAY_MOTION: 기존 JSON motion 이름 실행
- STOP
- RESET_EMERGENCY: 현장 안전 확인 후 latch 해제

MOVE_COORDS 좌표 단위는 [mm, mm, mm, deg, deg, deg]입니다. 위치/회전 범위,
NaN, 배열 길이를 먼저 검사합니다. 명령 직후 성공하지 않고 GET_COORDS를 polling해
position_tolerance_mm와 rotation_tolerance_deg 안에 들어온 뒤에만 reached true를
반환합니다.

성공 예:

    {"version":1,"request_id":"id","status":"ok","data":{"reached":true}}

오류 예:

    {"version":1,"request_id":"id","status":"error","error":{"code":"BUSY","message":"another command is in progress"}}

## 오류 코드

INVALID_JSON, INVALID_REQUEST, UNKNOWN_COMMAND, INVALID_COORDS,
COORDS_OUT_OF_RANGE, INVALID_SPEED, INVALID_MODE, BUSY, TARGET_TIMEOUT,
STOPPED, EMERGENCY_ACTIVE, ROBOT_READ_FAILED, EXECUTION_FAILED를 사용합니다.

## timeout

- TCP 연결 기본 3초
- 팔 명령 기본 60초
- MOVE_COORDS 도달 기본 30초
- client socket 기본 65초
- 타이어 검출 기본 10초

timeout은 성공으로 간주하지 않으며 같은 명령의 자동 중복 전송을 금지합니다.

## emergency 정책

STOP은 busy lock을 기다리지 않고 즉시 cancel event와 로봇 stop을 호출합니다.
STOP 뒤 emergency latch가 설정되어 신규 이동/그리퍼/motion 명령은
EMERGENCY_ACTIVE로 거부됩니다. GET_COORDS와 GET_ANGLES만 진단용으로 허용합니다.
현장 안전 확인 후 idle 상태에서만 RESET_EMERGENCY를 사용합니다. 로봇팔 비전
emergency manager는 좌우 STOP을 병렬 fan-out하고 새 side cycle을 차단합니다.


## Pinky route TCP (port 7000)

중앙제어 PC가 Pinky Raspberry Pi 192.168.0.98:7000에 연결합니다. 요청과 응답은
UTF-8 newline-delimited JSON이고 version은 1입니다.

공통 요청:

    {"version":1,"request_id":"unique-id","command":"COMMAND"}

지원 명령:

- PING: 연결 및 현재 외부 상태 확인
- GET_STATUS: state, current_station, station_index, total_stations, error 확인
- START_ROUTE: route=tire_stop_1이면 1번 위치까지만 이동 후 정차, route=tire_stop_2이면 2번 위치까지만 이동 후 정차, route=after_tire_service이면 3→4→5번 위치를 순차 주행
- STOP: Nav2 goal cancel, docking/retry 중단, zero Twist, STOPPED 전환
- RESET: STOPPED 또는 ERROR에서 IDLE로만 전환하며 주행하지 않음

START_ROUTE 예:

    {"version":1,"request_id":"start-1","command":"START_ROUTE","route":"after_tire_service"}

상태는 IDLE, RUNNING, COMPLETED, STOPPED, ERROR입니다. RUNNING 중 다른
request_id의 START_ROUTE는 BUSY입니다. 같은 request_id와 동일한 payload는 캐시된
응답을 반환하여 route를 다시 시작하지 않습니다. 같은 request_id를 다른 payload에
재사용하면 REQUEST_ID_CONFLICT입니다.

일시적인 GET_STATUS 연결 실패는 adapter가 마지막 상태를 유지하고 경고를 남깁니다.
명령 요청 실패와 전체 중앙 이동 timeout은 ERROR이며 성공으로 바뀌지 않습니다.

Pinky 오류 코드는 INVALID_JSON, INVALID_REQUEST, UNKNOWN_COMMAND, UNKNOWN_ROUTE,
BUSY, RESET_REQUIRED, INVALID_STATE, REQUEST_ID_CONFLICT, EXECUTOR_TIMEOUT,
EXECUTION_FAILED를 사용합니다. 중앙 client의 CONNECTION_FAILED와 TIMEOUT은
성공으로 처리하지 않으며 START_ROUTE를 자동 재전송하지 않습니다.

TCP 연결이 끊겨도 이미 실행 중인 route 상태는 바뀌지 않습니다. 안전 정지는 별도의
새 TCP 연결로 STOP을 보내야 합니다. 중앙 adapter는 상태 조회 실패를 성공 또는 COMPLETED로 변환하지 않고 마지막
확인 상태를 유지합니다. 연결이 복구되면 다음 GET_STATUS 결과를 반영하며, 중앙의
전체 Pinky 이동 timeout은 별도로 계속 적용됩니다.
