# AGENTS.md

이 문서는 `robot_L` 저장소를 수정하는 개발자와 AI 코딩 에이전트가 따라야 할 작업 규칙을 정의합니다.

## 1. 프로젝트 범위

현재 저장소는 무인 타이어 정비 시스템의 **왼쪽 JetCobot 로봇팔** 제어를 담당합니다.

현재 고정 조건:

- ROS 2: Jazzy
- 기본 작업공간: `/home/jetcobot/tire_ws`
- 대상 로봇팔: 오른쪽
- TCP 포트: `5000`
- systemd 서비스: `tire-arm-server.service`
- 서비스 시작 스크립트: `/home/jetcobot/start_tire_arm_server.sh`

명시적인 요구가 없는 한 포트 `5000`을 다른 값으로 변경하지 마십시오.

---

## 2. 핵심 설계 원칙

1. YOLO가 로봇팔을 직접 연속 제어하지 않습니다.
2. YOLO 결과는 상태 판단 또는 모션 선택에 사용합니다.
3. 실제 로봇 동작은 검증된 JSON 모션 또는 명확히 제한된 좌표 보정을 통해 수행합니다.
4. TCP 서버는 명령 수신과 실행 상태 반환 역할을 담당합니다.
5. 하드웨어 안전보다 편의성이나 코드 단순성을 우선하지 않습니다.
6. 왼팔 설정과 왼팔 설정을 한 파일에 무분별하게 혼합하지 않습니다.

---

## 3. 저장소 구조 규칙

권장 구조:

```text
robot_L/
├── README.md
├── AGENTS.md
├── deploy/
│   └── systemd/
│       └── tire-arm-server.service
├── scripts/
│   └── start_tire_arm_server.sh
└── tire_arm_control/
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── config/
    ├── launch/
    ├── motions/
    └── tire_arm_control/
```

새 파일을 추가할 때는 역할에 따라 다음 위치를 사용합니다.

- ROS 2 Python 코드: `tire_arm_control/tire_arm_control/`
- launch 파일: `tire_arm_control/launch/`
- YAML 설정: `tire_arm_control/config/`
- 로봇 모션 JSON: `tire_arm_control/motions/`
- 운영 스크립트: `scripts/`
- systemd 배포 예시: `deploy/systemd/`
- 테스트: `test/` 또는 패키지 내부 테스트 디렉터리

---

## 4. 필수 문서화 규칙

모든 ROS 2 패키지에는 `README.md`가 있어야 합니다.

각 패키지 README에는 최소한 다음 내용을 포함합니다.

- 패키지 목적
- 노드 목록과 각 노드의 역할
- launch 파일 목록
- 실행 방법
- 구독 토픽
- 발행 토픽
- 서비스와 액션
- 주요 파라미터
- 설정 파일
- 테스트 방법
- 알려진 제한 사항

코드 변경으로 다음 항목이 바뀌면 루트 `README.md`도 함께 수정합니다.

- TCP 포트
- TCP 명령
- 노드 이름
- launch 파일
- 토픽 또는 서비스
- 모션 파일
- systemd 서비스
- 시작 스크립트
- 빌드 및 실행 절차

---

## 5. TCP 서버 규칙

### 현재 포트

왼쪽 로봇팔은 반드시 기본값 `5000`을 사용합니다.

서버 바인딩 권장값:

```text
0.0.0.0:5000
```

`127.0.0.1`에만 바인딩하면 중앙 제어 PC에서 접속할 수 없으므로 사용하지 않습니다.

### 현재 명령

- `HELP`
- `HOME`
- `REMOVE_BAD_TIRE`
- `SET_ARUCO`

새 명령 추가 시 반드시 다음을 수행합니다.

1. 명령 문자열 상수화
2. 입력값 검증
3. 모션 또는 처리 함수와 명확하게 매핑
4. 성공/실패 응답 정의
5. 중복 명령 처리 정책 정의
6. README 명령 표 수정
7. 테스트 추가

권장 응답 상태:

```text
ACK
BUSY
DONE
ERROR
```

가능하면 다음 형태의 구조화된 메시지로 확장합니다.

```json
{
  "command_id": "uuid-or-sequence",
  "command": "HOME",
  "status": "ACK"
}
```

---

## 6. systemd 운영 규칙

실제 서비스 파일 위치:

```text
/etc/systemd/system/tire-arm-server.service
```

실제 시작 스크립트 위치:

```text
/home/jetcobot/start_tire_arm_server.sh
```

저장소에는 재설치 가능한 복사본을 다음 위치에 유지하는 것을 권장합니다.

```text
deploy/systemd/tire-arm-server.service
scripts/start_tire_arm_server.sh
```

시작 스크립트의 마지막 ROS 2 실행 명령에는 `exec`를 사용합니다.

```bash
exec ros2 launch tire_arm_control arm_server.launch.py
```

백그라운드 실행 기호 `&`를 사용하지 않습니다. systemd가 프로세스를 직접 추적해야 하기 때문입니다.

서비스를 수정한 뒤 반드시 다음을 수행합니다.

```bash
sudo systemctl daemon-reload
sudo systemctl restart tire-arm-server.service
systemctl status tire-arm-server.service --no-pager
```

서비스가 반복 재시작되면 다음 로그부터 확인합니다.

```bash
journalctl -u tire-arm-server.service -n 100 --no-pager
```

---

## 7. 로봇 모션 안전 규칙

모션 JSON을 수정하거나 추가할 때:

1. 관절 개수가 장비 사양과 일치해야 합니다.
2. 각 관절각은 허용 범위 안이어야 합니다.
3. 속도 값은 장비가 허용하는 범위 안이어야 합니다.
4. 첫 시험은 낮은 속도로 수행합니다.
5. 각 단계 사이에 충분한 대기 시간을 둡니다.
6. 경로상 충돌 가능성을 직접 확인합니다.
7. 기존 정상 모션을 덮어쓰기 전에 백업합니다.
8. 실패 시 홈 또는 안전 위치로 복귀할 수 있어야 합니다.

하드웨어 연결 없이 검증할 수 있도록 모션 파싱과 관절 범위 검사는 별도 함수로 작성합니다.

코드에서 권장하는 검증 순서:

```text
JSON 스키마 확인
→ 관절 개수 확인
→ 관절 범위 확인
→ 속도 범위 확인
→ 실행 전 상태 확인
→ 모션 실행
→ 결과 상태 반환
```

---

## 8. 코드 작성 규칙

- Python 3 타입 힌트를 사용합니다.
- 네트워크, 파일, 하드웨어 예외를 구분해 처리합니다.
- `print()`보다 ROS 2 logger를 사용합니다.
- IP, 포트, 파일 경로를 코드에 중복 하드코딩하지 않습니다.
- 포트 기본값은 설정 파일 또는 ROS 파라미터에서 관리합니다.
- 긴 동작 함수는 명령 파싱, 검증, 실행, 응답으로 분리합니다.
- 공유 상태에는 lock 또는 명시적인 단일 실행 큐를 사용합니다.
- 로봇이 동작 중일 때 새 모션 명령을 동시에 실행하지 않습니다.
- 소켓 종료와 ROS 2 노드 종료가 정상적으로 처리되어야 합니다.

---

## 9. 변경 금지 또는 주의 항목

명시적인 요청 없이 다음 항목을 변경하지 않습니다.

- 왼팔 TCP 포트 `5000`
- 작업공간 경로 `/home/jetcobot/tire_ws`
- systemd 서비스 이름 `tire-arm-server.service`
- 기존 정상 동작 JSON의 관절값
- 기존 명령 문자열
- 실제 장비 연결 설정

다음 작업은 특별히 주의합니다.

- 오른팔 포트 `5001` 설정을 왼팔 코드에 복사
- 모션 JSON의 클래스 또는 명령 매핑 반전
- YOLO 클래스 순서 변경
- 서버가 이미 동작 중인데 중복 서버 실행
- systemd 프로세스 내부에서 다시 백그라운드 실행
- 장치 허용 범위를 벗어난 관절각 실행

---

## 10. 변경 전 확인 절차

코드를 수정하기 전에 다음 파일을 우선 확인합니다.

```text
README.md
AGENTS.md
tire_arm_control/config/arm_config.yaml
tire_arm_control/launch/arm_server.launch.py
tire_arm_control/setup.py
tire_arm_control/package.xml
tire_arm_control/motions/
```

TCP 관련 변경 시 다음을 검색합니다.

```bash
grep -RIn "5001\|socket\|bind\|listen\|accept" .
```

명령 관련 변경 시:

```bash
grep -RIn "HELP\|HOME\|REMOVE_BAD_TIRE\|SET_ARUCO" .
```

---

## 11. 빌드 및 정적 확인

변경 후 기본 빌드:

```bash
cd /home/jetcobot/tire_ws
source /opt/ros/jazzy/setup.bash

colcon build \
  --symlink-install \
  --packages-select tire_arm_control

source install/setup.bash
```

Python 문법 확인:

```bash
python3 -m compileall src/tire_arm_control
```

패키지 확인:

```bash
ros2 pkg executables tire_arm_control
ros2 pkg prefix tire_arm_control
```

launch 파일 확인:

```bash
ros2 launch tire_arm_control arm_server.launch.py --show-args
```

---

## 12. 실행 및 통신 테스트

서비스 없이 수동 실행:

```bash
source /opt/ros/jazzy/setup.bash
source /home/jetcobot/tire_ws/install/setup.bash
ros2 launch tire_arm_control arm_server.launch.py
```

라즈베리파이에서 포트 확인:

```bash
ss -lntp | grep 5000
```

외부 PC에서 연결 확인:

```bash
nc -vz -w 3 <LEFT_ARM_IP> 5000
```

명령 테스트:

```bash
printf 'HELP\n' | nc <LEFT_ARM_IP> 5000
```

실제 모션 명령 테스트는 작업 반경을 비우고 낮은 속도에서 수행합니다.

---

## 13. 완료 조건

변경 작업은 다음 조건을 만족해야 완료된 것으로 봅니다.

- 패키지가 정상 빌드됨
- launch 파일이 로드됨
- TCP 포트 `5000`이 열림
- 외부 PC에서 연결 가능
- 잘못된 명령이 안전하게 거부됨
- 로봇 동작 중 중복 실행이 차단됨
- systemd 서비스가 `active (running)` 상태임
- README가 실제 코드와 일치함
- 새 기능에 대한 테스트 방법이 기록됨

---

## 14. 커밋 규칙

한 커밋에는 하나의 논리적 변경만 포함합니다.

권장 커밋 메시지:

```text
docs: add right arm setup guide
feat: add SET_ARUCO command
fix: keep TCP server process alive under systemd
refactor: separate command parser from motion executor
test: add invalid command handling test
```

모션 데이터 변경은 일반 코드 변경과 분리하는 것을 권장합니다.

```text
motion: update right arm observe pose
```

---

## 15. 에이전트 응답 규칙

AI 에이전트는 작업 결과를 설명할 때 다음을 명시합니다.

- 수정한 파일
- 변경 목적
- 포트 또는 명령 변경 여부
- 빌드 결과
- 테스트 결과
- 실제 로봇에서 추가 확인할 항목

실제 하드웨어 테스트를 수행하지 않았다면 성공했다고 단정하지 말고, 정적 검증 또는 소프트웨어 테스트까지만 완료했다고 정확히 기록합니다.
