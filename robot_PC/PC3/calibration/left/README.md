# Left arm calibration required

좌측은 우측과 동일한 파일 구조를 사용하지만 우측 보정값을 복사하지 않습니다.
실기 보정 후 아래 파일을 이 디렉터리에 넣으십시오.

- camera_calibration.npz
- handeye_calibration.npz
- slot_registration.json
- tire_grasp_model.json

파일이 없거나 키가 잘못되면 arm_aruco_perception이 명확한 오류를 발생시키고
MOVE_COORDS 전송을 차단합니다.
