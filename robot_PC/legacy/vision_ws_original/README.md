# 타이어 검출 PC

이 워크스페이스는 중앙 PC에서 TCP 명령을 수신하고, 좌우 타이어를 검출한 뒤 JSON 결과를 반환한다. 로봇팔을 제어하거나 중앙 PC 기능을 수행하지 않는다.

검출 PC의 실제 IP는 **192.168.0.81**, 서버 bind 주소는 **0.0.0.0**, 포트는 **6000**이다. 실제 IP를 bind 주소로 사용하지 않는다.

## 빌드 및 실행

```bash
cd /home/soo/robot_PC/vision_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 launch tire_vision vision_server.launch.py \
  mock_mode:=true tcp_host:=0.0.0.0 tcp_port:=6000
```

기본값은 `mock_mode=true`다. 카메라와 YOLO 모델이 없어도 left=`bad_tire`(0.91), right=`good_tire`(0.88), votes/frames=3인 응답을 반환한다.

## 검출 PC 내부 확인

```bash
ss -lntp | grep 6000
echo '{"version":1,"type":"health","request_id":"health-local-001"}' | nc -w 5 127.0.0.1 6000 | jq
echo '{"version":1,"type":"detect_tires","request_id":"detect-local-001"}' | nc -w 15 127.0.0.1 6000 | jq
```

## 중앙 PC 확인

```bash
nc -vz -w 3 192.168.0.81 6000
echo '{"version":1,"type":"health","request_id":"health-pc1-001"}' | nc -w 5 192.168.0.81 6000 | jq
echo '{"version":1,"type":"detect_tires","request_id":"detect-pc1-001"}' | nc -w 15 192.168.0.81 6000 | jq
```

방화벽은 검출 PC에서 `sudo ufw status`로 확인하고, 필요한 경우 운영 정책에 따라 TCP 6000 인바운드를 허용한다.

## 실제 장비 설정

설정 파일은 `src/tire_vision/config/vision.yaml`이다. `left_camera`와 `right_camera`에 각 카메라 index/해상도/FPS를 지정하고, `model_path`에 YOLO weights 경로를 지정한다. GPU는 `device: 0`, CPU는 `device: cpu`를 사용한다.

현재 검증 범위는 mock 모드다. 실제 좌우 카메라 연결·재연결과 실제 YOLO weights의 클래스 매핑 및 GPU/CPU 추론은 장비에서 추가 검증해야 한다.
