# Pinky 중앙제어 연동 시퀀스

## Pinky route 이름

- `tire_stop_1`: 1번 위치까지만 이동 후 `COMPLETED` 상태로 정차 유지
- `tire_stop_2`: 2번 위치까지만 이동 후 `COMPLETED` 상태로 정차 유지
- `after_tire_service`: 3번 -> 4번 -> 5번 위치를 순차 주행 후 `COMPLETED`

중앙제어 ROS topic 명령은 기존 형식을 유지한다.

- `START:tire_stop_1`
- `START:tire_stop_2`
- `START:after_tire_service`
- `STOP`
- `RESET`

TCP 브리지는 위 topic 명령을 Pinky TCP 7000의 `START_ROUTE` / `STOP` / `RESET` JSON 명령으로 변환한다.

## 교체 시퀀스

1. 중앙제어가 차량 주차/정차 완료를 확인한다.
2. 타이어 비전 결과로 교체 계획을 만든다. 좌/우 모두 BAD이면 left -> right 순서다.
3. left 교체 시작 시 `START:tire_stop_1`과 right arm `HELP`를 거의 동시에 시작한다.
4. left arm `REMOVE_BAD_TIRE`를 실행한다.
5. Pinky가 1번 위치에 도착했다는 `COMPLETED`가 확인되고 REMOVE도 끝났을 때 left arm `SET_ARUCO`를 실행한다.
6. left removed tire는 slot 3, new tire는 slot 4를 사용한다.
7. `INSTALL_NEW_TIRE` 후 left 완료 처리한다.
8. right가 필요한 경우 `START:tire_stop_2`와 left arm `HELP`를 시작하고 같은 순서로 진행한다.
9. right removed tire는 slot 1, new tire는 slot 2를 사용한다.
10. 필요한 모든 타이어 교체 후 양팔 HOME 완료 시 `START:after_tire_service`를 보내 3 -> 4 -> 5 순으로 이동한다.
11. 마지막 Pinky `COMPLETED`를 받으면 전체 cycle을 `COMPLETED`로 종료한다.

## 현재 남은 인터페이스 확인 사항

중앙제어 `RosTransportOperationHandler`는 아래 서비스를 기대한다.

- `/left_arm/transport_task`
- `/right_arm/transport_task`
- service type: `robot_interfaces/srv/TransportTask`

현재 PC3의 `left_arm_transport_pick` / `right_arm_transport_pick` 소스는 각각 아래 Trigger 서비스만 노출한다.

- `/left_arm/pick_transport_slot`
- `/right_arm/pick_transport_slot`
- service type: `std_srvs/srv/Trigger`

따라서 실제 slot 3/4/1/2 내려놓기/집기를 중앙제어에서 자동 호출하려면 이 서비스 규격을 맞추는 추가 작업이 필요하다.
