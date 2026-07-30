# robot_L

무인 타이어 정비 시스템의 **왼쪽 JetCobot 로봇팔 제어 저장소**입니다.

이 저장소는 ROS 2 Jazzy 환경에서 동작하며, 중앙 제어 PC로부터 TCP 명령을 수신해 미리 저장된 로봇팔 모션을 실행하는 구조를 사용합니다.

> 저장소 확인 참고
>
> GitHub 저장소가 외부 조회에서 공개되지 않아 실제 파일 트리는 직접 검증하지 못했습니다.
> 아래 구조와 파일명은 현재 프로젝트에서 사용 중인 `tire_arm_control` 패키지 구성을 기준으로 작성되었습니다.
> 저장소의 실제 파일명과 다르면 문서를 실제 코드에 맞게 수정해야 합니다.

> 현재 기준
>
> - 대상 로봇팔: 왼쪽 로봇팔
> - TCP 포트: `5000`
> - ROS 2 배포판: Jazzy
> - 기본 작업공간: `/home/jetcobot/tire_ws`
> - systemd 서비스: `tire-arm-server.service`

---

## 1. 시스템 역할

왼쪽 로봇팔은 중앙 제어 PC 또는 YOLO 판단 노드로부터 명령을 받아 다음 작업을 수행합니다.

1. 타이어 관찰 위치로 이동
2. 타이어 상태 판별 결과에 따라 저장된 모션 실행
3. 불량 타이어 탈거
4. ArUco 마커 기반 보정 동작 준비
5. 홈 위치 복귀
6. 테스트 및 비상 보조 동작 실행

YOLO가 로봇팔 관절을 직접 제어하는 방식이 아니라, YOLO 및 ArUco에서 계산된 판단 결과를 이용해 **기존에 검증된 로봇팔 모션을 선택하거나 목표 위치를 보정하는 방식**을 사용합니다.

---

## 2. 전체 시스템에서의 위치

```text
[탑다운 카메라 / 중앙 제어 PC]
    └─ 차량 주차 및 정지 상태 판단
            │
            ├─ TCP 명령 전송
            │
            └───────────────┐
                            │
                 [왼쪽 JetCobot]
                 IP: 현장 네트워크에서 확인
                 TCP Port: 5000
                            │
                 [tire_arm_control]
                            │
                 [저장된 JSON 모션 실행]
```

그리퍼 카메라에서 수행되는 YOLO 또는 ArUco 처리는 구성에 따라 별도 PC에서 실행될 수 있습니다. 이 저장소의 핵심 역할은 **왼쪽 로봇팔의 TCP 명령 수신과 모션 실행**입니다.

---

## 3. 현재 저장소 구조

현재 프로젝트에서 사용하는 기본 구조는 다음과 같습니다.

```text
robot_L/
├── README.md
├── AGENTS.md
└── tire_arm_control/
    ├── package.xml
    ├── setup.py
    ├── setup.cfg
    ├── resource/
    │   └── tire_arm_control
    ├── launch/
    │   └── arm_server.launch.py
    ├── config/
    │   └── arm_config.yaml
    ├── motions/
    │   ├── home.json
    │   ├── observe_tire.json
    │   ├── remove_bad_tire.json
    │   └── help.json
    └── tire_arm_control/
        ├── __init__.py
        └── ...
```

실제 파일명이 다르거나 파일이 추가된 경우, 이 구조도와 아래 파일 설명을 함께 갱신해야 합니다.

---

## 4. 주요 구성 요소

### `launch/arm_server.launch.py`

왼쪽 로봇팔 TCP 서버와 관련 ROS 2 노드를 실행하는 launch 파일입니다.

기본 실행 예시:

```bash
ros2 launch tire_arm_control arm_server.launch.py
```

왼쪽 로봇팔은 TCP 포트 `5000`을 사용하도록 설정해야 합니다.

### `config/arm_config.yaml`

로봇팔 연결 정보, TCP 포트, 모션 파일 경로, 속도 등 실행 설정을 관리합니다.

예시:

```yaml
tcp:
  host: 0.0.0.0
  port: 5000
```

실제 파라미터 이름은 현재 구현 코드와 일치시켜야 합니다.

### `motions/*.json`

JetCobot에서 실행할 관절 동작을 순서대로 저장합니다.

현재 사용 중인 대표 모션:

| 파일 | 역할 |
|---|---|
| `home.json` | 초기 위치 또는 안전 위치로 복귀 |
| `observe_tire.json` | 타이어 상태를 관찰할 위치로 이동 |
| `remove_bad_tire.json` | 불량 타이어 탈거 동작 |
| `help.json` | 통신 및 동작 확인용 테스트 모션 |

JSON 모션을 수정할 때는 반드시 다음 항목을 확인합니다.

- 모든 관절각이 JetCobot 허용 범위 안에 있는지
- 급격한 관절 변화가 없는지
- 로봇팔, 차량, 테이블과 충돌하지 않는지
- 낮은 속도에서 먼저 테스트했는지
- 각 단계 사이의 대기 시간이 충분한지

---

## 5. TCP 명령

현재 서버에서 사용하는 명령은 구현 상태에 따라 다음과 같이 구성됩니다.

| 명령 | 동작 |
|---|---|
| `HELP` | 서버 연결 및 테스트 동작 확인 |
| `HOME` | 홈 위치로 복귀 |
| `REMOVE_BAD_TIRE` | 불량 타이어 탈거 모션 실행 |
| `SET_ARUCO` | ArUco 기반 보정 또는 관련 단계 시작 |

추가 명령을 구현하면 다음 항목을 함께 수정해야 합니다.

1. TCP 명령 파서
2. ROS 2 노드 또는 모션 매핑
3. `motions/`의 JSON 파일
4. `README.md` 명령 표
5. `AGENTS.md` 유지보수 규칙
6. 테스트 명령 및 예상 응답

---

## 6. 빌드

라즈베리파이에서 다음 명령을 실행합니다.

```bash
cd /home/jetcobot/tire_ws

source /opt/ros/jazzy/setup.bash

colcon build --symlink-install

source /home/jetcobot/tire_ws/install/setup.bash
```

새 터미널을 열 때마다 ROS 2 환경과 작업공간을 다시 source해야 합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /home/jetcobot/tire_ws/install/setup.bash
```

---

## 7. 수동 실행

systemd를 사용하지 않고 직접 서버를 실행할 때:

```bash
cd /home/jetcobot/tire_ws

source /opt/ros/jazzy/setup.bash
source /home/jetcobot/tire_ws/install/setup.bash

ros2 launch tire_arm_control arm_server.launch.py
```

포트가 열렸는지 라즈베리파이에서 확인:

```bash
ss -lntp | grep 5000
```

다른 PC에서 연결 확인:

```bash
nc -vz -w 3 <왼쪽_로봇팔_IP> 5000
```

TCP 명령 전송 테스트:

```bash
printf 'HELP\n' | nc <왼쪽_로봇팔_IP> 5000
```

또는:

```bash
printf 'HOME\n' | nc <왼쪽_로봇팔_IP> 5000
```

---

## 8. 부팅 후 TCP 서버 자동 실행

TCP 서버를 계속 열어두기 위해 라즈베리파이의 systemd에 서비스를 등록해 두었습니다.

이 설정 파일은 일반적으로 저장소 내부가 아니라 라즈베리파이의 다음 경로에 존재합니다.

```text
/etc/systemd/system/tire-arm-server.service
```

현재 기준 서비스 이름:

```text
tire-arm-server.service
```

현재 기준 시작 스크립트:

```text
/home/jetcobot/start_tire_arm_server.sh
```

### 8.1 시작 스크립트 예시

`/home/jetcobot/start_tire_arm_server.sh`

```bash
#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source /home/jetcobot/tire_ws/install/setup.bash

exec ros2 launch tire_arm_control arm_server.launch.py
```

실행 권한 부여:

```bash
chmod +x /home/jetcobot/start_tire_arm_server.sh
```

### 8.2 systemd 서비스 예시

`/etc/systemd/system/tire-arm-server.service`

```ini
[Unit]
Description=Left Tire Arm TCP Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetcobot
WorkingDirectory=/home/jetcobot/tire_ws
ExecStart=/home/jetcobot/start_tire_arm_server.sh
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

> 이 서비스는 왼쪽 로봇팔의 TCP 서버를 실행하며, 애플리케이션 설정상 포트 `5000`을 사용해야 합니다.

### 8.3 서비스 적용

```bash
sudo systemctl daemon-reload
sudo systemctl enable tire-arm-server.service
sudo systemctl restart tire-arm-server.service
```

### 8.4 서비스 상태 확인

```bash
systemctl status tire-arm-server.service --no-pager
```

실시간 로그 확인:

```bash
journalctl -u tire-arm-server.service -f
```

최근 로그 확인:

```bash
journalctl -u tire-arm-server.service -n 100 --no-pager
```

자동 시작 등록 여부 확인:

```bash
systemctl is-enabled tire-arm-server.service
```

포트 확인:

```bash
ss -lntp | grep 5000
```

### 8.5 서비스가 반복 재시작되는 경우

상태가 다음처럼 보이면 서버 프로세스가 실행 직후 종료된 것입니다.

```text
Active: activating (auto-restart)
```

다음 순서로 확인합니다.

```bash
journalctl -u tire-arm-server.service -n 100 --no-pager

bash -x /home/jetcobot/start_tire_arm_server.sh

source /opt/ros/jazzy/setup.bash
source /home/jetcobot/tire_ws/install/setup.bash
ros2 launch tire_arm_control arm_server.launch.py
```

시작 스크립트 마지막 명령에는 `exec`를 사용해 실제 ROS 2 프로세스가 systemd의 메인 프로세스로 유지되도록 합니다.

---

## 9. 네트워크 확인

라즈베리파이 IP 확인:

```bash
hostname -I
```

왼쪽 로봇팔의 서버는 모든 인터페이스에서 접속할 수 있도록 일반적으로 다음 주소에 바인딩합니다.

```text
0.0.0.0:5000
```

확인:

```bash
sudo ss -lntp | grep ':5000'
```

중앙 제어 PC와 로봇팔이 같은 네트워크에 있고 서로 ping 가능한지 확인합니다.

```bash
ping -c 4 <왼쪽_로봇팔_IP>
```

---

## 10. 안전 주의사항

로봇팔 동작을 실행하기 전에 반드시 다음을 확인합니다.

- 작업 반경 안에 사람이 없는지
- 비상 정지 또는 전원 차단 수단을 즉시 사용할 수 있는지
- 처음 실행하는 모션은 낮은 속도로 검증했는지
- JSON 관절값이 장비의 허용 범위를 넘지 않는지
- 케이블이 로봇팔에 걸리지 않는지
- 서버 명령이 중복 수신되어 모션이 재실행되지 않는지
- 네트워크 재연결 후 이전 명령이 다시 처리되지 않는지

실제 하드웨어에서 테스트하지 않은 동작을 바로 최대 속도로 실행하지 않습니다.

---

## 11. 문제 해결

### TCP 연결 거부

```bash
systemctl status tire-arm-server.service --no-pager
ss -lntp | grep 5000
journalctl -u tire-arm-server.service -n 100 --no-pager
```

확인 항목:

- 서비스가 실행 중인지
- 서버가 `127.0.0.1`이 아닌 `0.0.0.0`에 바인딩됐는지
- 포트가 실제로 `5000`인지
- 방화벽에서 `5001/tcp`가 차단되지 않았는지
- 중앙 PC와 왼팔 라즈베리파이의 IP 대역이 같은지

### 패키지를 찾지 못함

```bash
source /opt/ros/jazzy/setup.bash
source /home/jetcobot/tire_ws/install/setup.bash
ros2 pkg list | grep tire_arm_control
```

필요하면 다시 빌드합니다.

```bash
cd /home/jetcobot/tire_ws
rm -rf build/tire_arm_control install/tire_arm_control log
colcon build --symlink-install --packages-select tire_arm_control
source install/setup.bash
```

### 관절각 범위 오류

모션 JSON의 각 관절값이 JetCobot의 허용 범위를 벗어났는지 확인합니다. 허용 범위를 넘는 값은 실행 전에 차단하고 로그에 어느 관절이 문제인지 출력하도록 구현하는 것을 권장합니다.

---

## 12. 향후 추가 권장 사항

- TCP 요청과 응답 형식을 JSON 메시지로 표준화
- 명령 ID를 추가해 중복 명령 실행 방지
- `BUSY`, `DONE`, `ERROR` 상태 응답 구현
- 비상 정지 명령 추가
- 모션 실행 전 관절 범위 자동 검증
- 왼팔/오른팔 설정 파일 분리
- systemd 설정 예시를 `deploy/systemd/`에 복사본으로 보관
- 시작 스크립트를 `scripts/`에 저장하고 설치 절차 문서화
- 각 ROS 2 노드의 토픽, 서비스, 파라미터를 README에 추가
- 자동 테스트 또는 모의 실행 모드 추가

---

## 13. 권장 배포 파일 구조

systemd 원본은 `/etc/systemd/system/`에 있어야 하지만, 재설치와 팀 공유를 위해 저장소에도 예시 파일을 보관하는 것을 권장합니다.

```text
robot_L/
├── deploy/
│   └── systemd/
│       └── tire-arm-server.service
├── scripts/
│   └── start_tire_arm_server.sh
├── README.md
└── AGENTS.md
```

저장소의 파일을 실제 시스템에 설치하는 예시:

```bash
sudo cp deploy/systemd/tire-arm-server.service \
  /etc/systemd/system/tire-arm-server.service

sudo cp scripts/start_tire_arm_server.sh \
  /home/jetcobot/start_tire_arm_server.sh

sudo chmod +x /home/jetcobot/start_tire_arm_server.sh
sudo systemctl daemon-reload
sudo systemctl enable --now tire-arm-server.service
```

---

## 14. 문서 유지 규칙

코드 또는 설정을 변경할 때 다음 내용을 반드시 함께 갱신합니다.

- 실행 명령
- TCP 포트와 명령 목록
- ROS 2 노드, 토픽, 서비스, 파라미터
- 모션 JSON 파일 목록
- systemd 서비스와 시작 스크립트
- 테스트 절차
- 알려진 제한 사항

세부 개발 규칙은 [`AGENTS.md`](./AGENTS.md)를 참고합니다.
