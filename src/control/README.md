# control — MORAI UDP 차량 제어

대회에서 SIM 통신 전체를 UDP로 요구하면 별도
`unita_launch/competition_udp_tracking.launch`를 사용합니다.
새 `scripts/competition_udp_control.py`는 ROS Bridge/모드 조회 없이
공식 예제의 AutoMode=2를 UDP 요청하고, EgoStatus UDP를 수신합니다.
기존 아래 ROS 서비스 기반 동작은 그대로 유지합니다.
GPS + EgoStatus heading이 기본 입력이며 실제 IMU는 옵션입니다.
실행·GUI 포트·중지 방법은 [대회용 UDP 전용 주행](../unita_launch/README.md#대회용-udp-전용-주행)을
참고하세요. 새 packet encoder는 기존 encoder의 검증/조향 정규화를 재사용합니다.

`/ctrl_cmd` (`morai_msgs/CtrlCmd`)를 받아 MORAI **Ego Ctrl Cmd** 55-byte UDP
패킷으로 변환하고 SIM의 Cmd Control 수신 주소로 20 Hz 송신합니다.
기존 waypoint follower의 `/ctrl_cmd`를 사용하며 센서 UDP와 차량 제어 UDP는 별개입니다.

## MORAI 설정

1. Network Settings → Cmd Control에서 Ego 차량 제어를 UDP로 설정합니다.
   Ghost 위치 이동 제어가 아닌 일반 Ego Ctrl Cmd / MoraiCmdController를 사용합니다.
2. SIM에서 명령을 받는 IP/포트와 아래 `morai_ip`, `morai_port`를 맞춥니다.
   현재 사용자 SIM에서 확인한 Cmd Control Host는 `127.0.0.1:9093`입니다.
   센서 Destination 포트(9201 등)로 제어 명령을 보내면 안 됩니다.
3. **Service는 ROS로 유지**하고 `/Service_MoraiEventCmd`가 연결되어 있어야 합니다.
   기본 모드 조회는 이 서비스를 `option=0`으로 호출합니다. 제어 모드·기어·램프·
   pause를 변경하지 않고 현재 상태만 읽습니다. **Ego Vehicle Status Publisher는
   ROS로 유지할 수 있으며 상태 UDP 9092 설정은 기본 실행에 필요 없습니다.**
4. Connect 후 Q로 화면의 차량 모드가 **AV / ExternalCtrl**인지 확인합니다.
   사용자 SIM의 서비스 응답은 수동 `1`, 외부 제어 `3`, 내장 자율주행 `6`입니다.
   서비스 외부 제어 `3`을 내부 게이트 `2`로 변환해 송신을 허용합니다.
   UDP CtrlMode는 현재 모드를 유지하는 `0`, gear는 `4`(D)입니다.
   `/control/sim_mode`는 변환된 값(수동 1, 외부 2, 그 외 0)입니다.
5. 기존 ROS 차량 제어 연결은 동시에 사용하지 않습니다. 센서만 UDP로 바꾸어도
   차량 제어 연결은 자동으로 UDP로 바뀌지 않습니다.

SIM이 별도 Windows PC에서 실행 중이면 `127.0.0.1`은 사용할 수 없습니다.
SIM PC의 LAN IPv4를 지정하세요. SIM에서 지정한 수신 포트의 UDP가 방화벽에서
허용되어 있어야 합니다. 송신 PC의 소스 포트를 고정해야 하는 설정에서는
`config/morai_udp.yaml`의 `local_port`를 지정합니다(기본 0: OS 자동 선택).

## 빌드 및 실행

```bash
source ~/catkin_ws/devel/setup.bash
cd ~/unita_ws
catkin_make
source devel/setup.bash
```

먼저 sensor_bridge를 별도 터미널에서 실행합니다. 현재 사용자에게 확인된 지도
offset은 아래 값입니다. 지도가 바뀌면 다시 확인해야 합니다.

```bash
roslaunch sensor_bridge sensor_bridge.launch \
  east_offset:=302595.0 north_offset:=4124145.0
```

새 터미널에서 **주행 + UDP 송신**을 실행합니다. 현재처럼 SIM과 ROS가 같은 PC이면
아래 주소를 사용합니다. 별도 PC이면 `morai_ip`에 실제 SIM PC IP를 지정합니다.

```bash
source ~/unita_ws/devel/setup.bash
roslaunch control waypoint_udp.launch morai_ip:=127.0.0.1 morai_port:=9093
```

**RViz까지 함께 실행**하려면 위 launch 대신 다음을 사용합니다.

```bash
roslaunch unita_launch waypoint_udp_tracking.launch \
  morai_ip:=127.0.0.1 morai_port:=9093
```

위 두 launch는 기존 `gps_waypoint.launch`를 포함합니다. 이미 waypoint follower가
실행 중이면 중복으로 켜지 말고, **UDP 송신 노드만** 추가하세요.

```bash
roslaunch control morai_udp_control.launch morai_ip:=127.0.0.1 morai_port:=9093
```

## 조향 및 입력 처리

ROS steering은 rad, UDP steer는 −1~1입니다.

```text
UDP steer = clamp(steering_sign × ROS steering / radians(max_steering_deg), −1, 1)
```

기본 `max_steering_deg=36.25`는 MORAI 문서의 Niro/Ioniq 예시값입니다.
실제 차량 Dynamics의 최대 조향각으로 조정합니다.
기존 follower의 `max_steering_rad=0.5`는 제어 출력 제한이고 차량 최대 조향각과는
다른 값입니다. 부호가 반대이면 송신 설정의 `steering_sign`을 조정하되 follower의
steering_sign까지 동시에 반전하지 마세요.

```bash
roslaunch control morai_udp_control.launch \
  morai_ip:=127.0.0.1 morai_port:=9093 max_steering_deg:=36.25
```

- 속도 단위는 km/h이며 기본 최대 송신 목표 속도는 30 km/h입니다.
- CtrlCmd longlCmdType 1/2/3을 매핑합니다. 기본 follower는 2(속도 제어)를 사용합니다.
- 속도 목표가 0이면 1(페달 제어)로 전환해 accel=0, brake=1, steer=0을 송신합니다.
- 새 명령이 0.5초간 없거나 GPS/IMU가 1초간 오래되면 같은 제동 패킷을 송신합니다.
  GPS no-fix, 잘못된 quaternion/명령, 미래 timestamp도 제동 상태로 전환합니다.
- sensor_bridge GPS status는 NMEA fix quality로 해석합니다. GPS가 다른 publisher라면
  status 의미를 확인해야 합니다. 센서 timestamp와 ROS 시간 기준이 일치해야 합니다.
- 송신 루프는 wall clock을 사용해 /clock이 멈춰도 timeout을 검사합니다.
- 종료 시 신선한 상태가 외부 제어 모드일 때만 제동 패킷을 3회 송신합니다. UDP에는 ACK가 없으므로 실제 적용을 보장하는
  응답은 아닙니다. 프로세스 강제 종료, 네트워크 단절은 최종 제동 전달도 막을 수 있습니다.

## 수동 전환 및 재개

- Q로 수동 모드로 바꾸면 모드 조회 응답을 받은 다음 송신 주기부터 모든 제어 송신을
  중단합니다. 수동 상태에서는 watchdog 제동과 종료 제동도 보내지 않습니다.
- 내장 자율주행 등 외부 제어 이외의 모드에서도 송신하지 않습니다.
- 외부 제어로 복귀하면 새 `/ctrl_cmd`를 받은 뒤 주행을 재개합니다.
  모드 전환 전 명령은 버립니다. follower의 계산 로그는 수동 모드에서도 나올 수
  있으므로 실제 송신 여부는 `/control/udp_status`로 판단합니다.
- 모드 조회 응답이 없거나 0.3초간 오래되면 `PAUSED`로 전환하고
  송신하지 않습니다. 모드를 모르므로 AutoMode 제동도 강제로 보내지 않습니다.
  따라서 상태 통신 단절 때 차량 정지까지 보장하지는 않습니다.
- 모드 판단에는 상태 전송/수신 지연이 있지만, 조회 사이에 도착하는 패킷도
  현재 모드를 유지하므로 Q로 선택한 수동 모드를 외부 제어로 되돌리지 않습니다.

## 선택적으로 상태 UDP를 사용하는 경우

`mode_source:=udp status_port:=9092`를 launch 인자로 전달하면 이전 상태 UDP
방식을 사용합니다. 이 경우에만 SIM Ego Vehicle Status Publisher를 UDP로 바꾸고
ROS PC의 Destination IP/Port 9092로 50 Hz 전송합니다. UDP 모드 값은 서비스와
다르며 수동 `1`, 외부 제어 `2`, 그 외 `0`입니다. 기본 `mode_source:=service`와
동시에 사용하지 않습니다.

## 확인

```bash
rostopic info /ctrl_cmd
rostopic echo -n 1 /ctrl_cmd
rostopic echo /control/udp_status
rostopic echo /control/sim_mode
```

`/ctrl_cmd` Subscribers에 `/morai_udp_control`이 있어야 합니다.
`SENDING`은 신선한 명령과 센서를 받아 UDP sendto가 성공한 상태입니다.
SIM 수신·기어·차량 움직임까지 확인하는 ACK는 아닙니다.
`PAUSED`는 수동/다른 모드이거나 상태 UDP가 없어 실제 명령을 보내지 않는 상태입니다.
시작부터 `PAUSED: missing or stale`이면 `/Service_MoraiEventCmd` 연결을 확인합니다.
`/control/sim_mode`는 마지막 수신 모드이므로 통신이 끊겼는지는 `udp_status`로 봅니다.
`BRAKE` 상태는 신선한 명령 또는 센서 조건을 만족하지 못한 것이므로 이유를 확인합니다.

모두 정상이지만 차가 움직이지 않으면 SIM의 Cmd Control 수신 IP/포트,
Connect 상태, ExternalCtrl 모드 및 실제 차량 기어를 확인합니다.

## 구성 및 검증

- `src/unita_control/udp_protocol.py`: 패킷 인코딩, 상태 해석, 서비스 모드 변환, 모드 게이트, 단위 변환, watchdog
- `scripts/morai_udp_control.py`: ROS 구독, 읽기 전용 서비스 모드 조회/선택적 상태 UDP 수신, 조건부 명령 UDP 송신, 상태 발행
- `config/morai_udp.yaml`: 토픽, 송신 주기, timeout, 속도 제한, 조향 부호
- `launch/morai_udp_control.launch`: 송신 노드만 실행
- `launch/waypoint_udp.launch`: 기존 waypoint follower + 송신 노드
- `test/test_udp_protocol.py`: 프로토콜/조향/제동/timeout 테스트 15개

공식 ctypes EgoCtrlCmd 예제와 생성 바이트가 일치하는지 비교했습니다.
별도 ROS master와 localhost UDP 수신기로 ROS→UDP 통합 테스트를 수행했습니다.
서비스 조회 option=0 및 수동/내장/외부 제어 순환을 검증했습니다.
수동/내장 모드에서 명령과 제동 송신이 없는지, 상태 단절/고정 timestamp에서
송신을 멈추는지, 외부 제어 복귀 및 수동 모드 종료 시 제동이 없는지도 확인했습니다.
모드 전환 개선은 실제 SIM에서 Q로 확인해야 합니다.

```bash
catkin_make run_tests_control
catkin_test_results build/test_results/control
```

참고: [공식 EgoCtrlCmd 정의](https://github.com/MORAI-Autonomous/MORAI-NetworkModule/blob/24.R2.0/lib/define/EgoCtrlCmd.py),
[MORAI UDP 통신 메시지](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1),
[ROS 이벤트 서비스](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/ros-2).
## Q 키 모드 전환

현재 설치된 MORAI `S4.251001.MolitComp03`에서는 제어 패킷의 CtrlMode가
`0`이면 실제 제어 모드를 유지합니다. 송신기는 이 값을 사용하고 Q 키로 선택한
모드를 변경하지 않습니다. 기존 `2`는 패킷마다 외부 제어를 다시 선택하므로,
Q로 수동 전환한 순간 다음 상태 조회 전에 패킷이 도착하면 수동 전환을 취소할
수 있었습니다. 일반 주행 명령과 watchdog/종료 브레이크 모두 모드 유지 값을
사용합니다. 이 값은 현재 설치 SIM의 수신 처리와 실제 수동 상태 조회로
검증하며, 다른 SIM 버전에서는 해당 버전의 처리 방식 확인이 필요합니다.

모드 확인은 기존 `/Service_MoraiEventCmd`를 `option=0`으로 조회하고,
외부 제어에서만 UDP를 송신합니다. Q 순서는 SIM의 기본
**수동 → 내장 자율주행 → 외부 waypoint 주행 → 수동**을 따릅니다.
수동 또는 내장 자율주행을 확인하면 브레이크 패킷도 송신하지 않습니다.
