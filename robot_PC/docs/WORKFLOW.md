# 전체 작업 흐름

    IDLE
      -> VEHICLE_DETECTED
      -> PARKING_STABLE
      -> DETECT_BOTH
         -> RECHECK(left 또는 right) -> DISCARD_PAIR -> DETECT_BOTH
         -> FINAL_PAIR
      -> BUILD_PLAN(left, right)
      -> LEFT_FULL_CYCLE (대상일 때)
      -> RIGHT_FULL_CYCLE (대상일 때)
      -> COMPLETED

SET_CAMERA 단계는 없습니다.

각 팔의 full cycle은 안정 ArUco 확인, 팔 전용 Hand-Eye 확인, 교체 필요 타이어
제거, 제거 타이어 운반차량 슬롯 배치, 새 타이어 집기, 새 타이어 장착, HOME 순입니다.
approach/grasp/place 좌표 중 하나라도 없으면 시작하지 않습니다.

## RECHECK

GOOD/BAD 경계는 운영 캡처 기반 radial gap ratio 0.28과 0.32입니다. 0.28 이하는 BAD,
0.32 이상은 GOOD, 그 사이는 RECHECK입니다. 판정은 픽셀 gap이 아니라 gap_px / radius_px를 사용합니다.
운영 안정화는 15개 ratio window에서 최소 10개 유효 측정과 표준편차 0.02 이하를 요구합니다.
마커 미검출은 연속 1~3프레임 동안 직전 안정 GOOD/BAD를 유지하고, 4~5프레임은 RECHECK,
6프레임부터 ERROR입니다. 안정 판정 전 초기 미검출은 RECHECK입니다. HSV [15,40,60]~[42,255,255], 최소 면적 5,
타이어 상단 ROI 1.15 확장을 사용합니다. 한쪽 결과가 최종이어도 다른 쪽이 RECHECK면
두 결과를 모두 폐기하고 reset_cycle=true로 좌우 전체를 다시 검출합니다.

## left-first

교체 계획은 항상 left, right 순서입니다. 양쪽이 BAD면 left 완료 flag가 설정되기
전에는 right start가 RuntimeError로 차단됩니다. 오른쪽만 BAD인 경우에는 바로
오른쪽을 시작할 수 있습니다.

## emergency

차량 움직임, 사용자 STOP, 통신 오류 또는 작업 오류가 발생하면 정상 worker 결과를
더 이상 적용하지 않고 좌우 STOP을 우선 전송합니다. latch 동안 신규 작업을 받지
않습니다. 현장 확인과 수동 reset 뒤에만 새 cycle을 시작합니다.
