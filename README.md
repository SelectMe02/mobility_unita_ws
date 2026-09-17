# UNITA MORAI workspace

ROS1 Noetic용 MORAI 센서 UDP 브릿지, GPS/IMU localization, waypoint 주행과
RViz 시각화 워크스페이스입니다. MORAI 메시지는 별도 `catkin_ws`에서 먼저
빌드한 다음 이 워크스페이스를 overlay로 빌드합니다.

처음 내려받을 때 워크스페이스 폴더 이름을 아래처럼 지정합니다.

```bash
git clone https://github.com/SelectMe02/catkin_ws.git ~/catkin_ws
git clone https://github.com/SelectMe02/mobility_unita_ws.git ~/unita_ws
```

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
cd ~/unita_ws
catkin_make
source ~/unita_ws/devel/setup.bash
```

MORAI 센서 Destination 포트와 `sensor_bridge/config/sensors.yaml`을 맞춥니다.
현재 K-city waypoint 기준 map offsets는 east `302595`, north `4124145`입니다.
맵을 바꾸면 해당 맵의 실제 offsets로 변경합니다.

```bash
roslaunch sensor_bridge sensor_bridge.launch \
  east_offset:=302595 north_offset:=4124145
```

별도 터미널에서 SIM의 Cmd Control 수신 IP/포트에 맞춰 실행합니다.
현재 SIM이 같은 컴퓨터에서 실행되며 수신 포트가 9093인 구성입니다.
Ego Network의 Service는 ROS `/Service_MoraiEventCmd`로 유지합니다.

```bash
source ~/unita_ws/devel/setup.bash
roslaunch unita_launch waypoint_udp_tracking.launch \
  morai_ip:=127.0.0.1 morai_port:=9093
```

`loop_path`의 기본값은 `true`입니다. 마지막 waypoint에 도착해도 정지하지
않고 첫 waypoint로 이어서 계속 주행합니다. 마지막 좌표가 첫 좌표와 같으면
추종 탐색에서 마지막 중복 좌표만 제외하고 RViz 경로 표시는 유지합니다.
nearest 탐색과 lookahead 탐색 모두 마지막/첫 지점 경계를 넘어갑니다.
단일 주행 후 정지가 필요하면 launch에 `loop_path:=false`를 지정합니다.
경로에 전방 목표가 없거나 UDP 송신기의 센서/명령 watchdog이 만료되면
기존 정지 처리가 적용됩니다. 수동/내장 자율주행 모드에서는 UDP 제어 송신을
중단합니다.

RViz의 경로·궤적·오차 분석, 카메라 보기와 I 키 초기 위치 설정은
[`unita_launch/README.md`](src/unita_launch/README.md)를 참고하세요.
Windows MORAI PC와 Ubuntu PC를 LAN으로 연결하는 현장 구성은
[`5분 현장점검 절차`](src/unita_launch/README.md#field-check--5분-현장점검-절차)를
먼저 따릅니다. 센서 Destination/ROS Bridge IP는 Ubuntu Ethernet IPv4,
제어 `morai_ip`는 Windows Ethernet IPv4입니다. 현장에서는 위 로컬 예제의
`127.0.0.1`을 사용하지 않습니다. `field_check.launch`는 진단과 센서 수신만
시작하며 차량 제어 노드는 시작하지 않습니다.
UDP 제어 설정은 [`control/README.md`](src/control/README.md),
센서 설정은 [`sensor_bridge/README.md`](src/sensor_bridge/README.md)에 있습니다.

```bash
catkin_make run_tests
catkin_test_results
```

Git에는 소스와 설정을 저장하며 `build`, `devel`, 로그, Python 캐시는 제외합니다.
