# 시스템 아키텍처

## 장비별 역할

| 장비 | 배포 디렉터리 | 역할 |
|---|---|---|
| 중앙제어 PC | central_ws | 차량 정지 확인, 양쪽 타이어 검출 요청, RECHECK 재시도, 교체 대상/개수 산정, 전체 비상정지 |
| 타이어 검출 PC | vision_ws | 카메라 2/4 입력, 단일 tire YOLO11 segmentation, 노란선 HSV, radial gap ratio 판정 |
| 로봇팔 비전 PC | arm_vision_ws + calibration | 좌우 ArUco 안정화, 팔별 Hand-Eye, 슬롯 approach/grasp/place 계산, left-first 작업 조정 |
| 왼쪽 Raspberry Pi | raspberry_left_ws | 왼팔/그리퍼 실물 제어, TCP 5000 |
| 오른쪽 Raspberry Pi | raspberry_right_ws | 오른팔/그리퍼 실물 제어, TCP 5001 |
| Pinky Raspberry Pi | pinky/roscamp-repo-2-pinky_a | Nav2 + 기존 정밀 도킹 + TCP route server 7000 |

로봇팔 비전 PC는 pymycobot을 import하지 않습니다. 실제 직렬 로봇 제어는 각
Raspberry Pi에만 존재합니다. 운반차량 자체 이동은 현재 범위에서 제외합니다.

## 데이터 흐름

1. 중앙제어가 상부 카메라에서 차량 정지를 확정합니다.
2. 중앙제어가 타이어 검출 PC TCP 6000에 양쪽 검출을 한 요청으로 보냅니다.
3. 어느 한쪽이라도 RECHECK면 해당 응답 전체를 버리고 양쪽 모두 재검출합니다.
4. 둘 다 GOOD/BAD로 확정되면 left, right 순서의 교체 계획과 개수를 만듭니다.
5. 로봇팔 비전은 대상 팔의 그리퍼 카메라 ArUco가 안정될 때까지 좌표를 만들지 않습니다.
6. 팔 전용 Hand-Eye와 슬롯 등록을 결합해 approach/grasp/place를 계산합니다.
7. 계산 좌표와 단계 명령을 해당 Raspberry Pi로 전송합니다.
8. 양쪽 대상이면 왼쪽의 제거-적재-새 타이어 집기-장착 전체가 끝난 뒤 오른쪽을 시작합니다.

## TCP 관계

| 송신 | 수신 | 포트 | 내용 |
|---|---|---:|---|
| 중앙제어 PC | 타이어 검출 PC | 6000 | health, detect_tires JSON; 기존 응답 필드 유지 |
| 로봇팔 비전 PC | 왼쪽 Raspberry Pi | 5000 | 기존 plain text 및 JSON arm command |
| 로봇팔 비전 PC | 오른쪽 Raspberry Pi | 5001 | 기존 plain text 및 JSON arm command |
| 중앙제어 PC | 좌우 팔 | 해당 연결 | 비상 시 STOP fan-out |
| 중앙제어 PC | Pinky Raspberry Pi | 7000 | PING/GET_STATUS/START_ROUTE/STOP/RESET JSON |

중앙제어는 SET_CAMERA를 호출하지 않습니다. SET_ARUCO 자세는 각 팔의 그리퍼
카메라 작업 자세이며 Raspberry Pi의 팔 전용 motion으로 관리합니다.


## Pinky 이동 구조

    Central PC / central_control
        |
        | /pinky/route_command compatibility topic
        v
    pinky_tcp_bridge
        |
        | TCP 192.168.0.98:7000
        v
    Pinky Raspberry Pi / pinky_tcp_server
        |
        | executor command queue
        v
    existing WaypointManager
        |
        +--> Nav2 NavigateToPose
        |
        +--> existing DockingStateMachine --> /cmd_vel --> Pinky Base

Pinky bringup은 hardware/scan/odom을 제공하지만 NavigateToPose action server를
제공하지 않습니다. pinky_navigation bringup이 기존 map, AMCL, controller,
planner, behavior server, BT navigator, velocity smoother를 올립니다. route server는
이들과 분리되어 부팅 후 IDLE로 기다립니다.

systemd 의존 관계는 pinky-bringup.service → pinky-navigation.service →
pinky-route.service입니다. 저장소 복사본에는 실제 운용 중인
pinky-bringup.service unit 원문이 포함되어 있지 않으므로 기존 unit은 수정하지 않고
새 두 unit의 템플릿만 제공합니다.
