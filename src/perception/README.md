# perception — 센서 인식

## 신호등 모델 테스트

`traffic_light_test.py`는 전방 카메라의 `sensor_msgs/Image`를 받아 `best1.pt`로
신호등을 감지하고 상태, 학습 클래스명, 신뢰도, bbox(x1, y1, x2, y2)를 ROS 로그로 출력합니다.
초록 / 초록+좌회전 / 좌회전 / 빨강 / 빨강+노랑 / 노랑을 구분합니다.
여러 신호등은 각각 출력하며 주행 대상 신호등 선정이나 차량 제어는 수행하지 않습니다.
미검출, 영상 수신 중단, 추론 오류도 별도로 출력합니다.

### 실행

모델의 학습 메타데이터에 기록된 Ultralytics 버전은 8.4.55입니다.
ROS와 cv_bridge를 사용할 수 있는 Python 환경에 의존성을 설치하세요.

```bash
python3 -m pip install ultralytics==8.4.55
cd /home/unita/unita_ws
catkin_make
source devel/setup.bash
# sensor_bridge와 시뮬레이터 카메라를 먼저 실행한 상태에서:
roslaunch perception traffic_light_test.launch
```

기본 모델 경로는 소스 워크스페이스 루트의 `best1.pt`입니다.
다른 경로나 install 공간에서 실행할 때는 절대 경로로 지정하세요.

```bash
roslaunch perception traffic_light_test.launch model_path:=/absolute/path/best1.pt image_topic:=/camera/image/front confidence:=0.5 device:=cpu
```

`device:=0`은 CUDA GPU 선택입니다. 기본값은 CPU입니다.
`inference_rate`는 초당 추론 상한(기본 2회), `image_timeout`은 영상 수신/처리 제한 시간(기본 2초)입니다.
CPU 추론이 느려 결과가 생략되면 `image_timeout:=5.0` 등으로 조절하세요.
항상 최신 수신 프레임을 사용하며 로그는 터미널 및 ROS 기본 로그(`~/.ros/log`)에서 확인합니다.

출력 예시(실제 실행 결과가 아닌 형식 예시):

```text
신호등 1개 | 빨강 (traffic_light_red) confidence=0.943 bbox=(310, 120, 355, 145)
신호등 미검출 (confidence >= 0.50)
```

추론 인터페이스: https://docs.ultralytics.com/modes/predict/
