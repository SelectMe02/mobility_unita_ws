# unita_launch — 주행 실행 및 RViz 시각화

기존 waypoint follower와 GPS+IMU localization을 조합해 기준 경로, 실제 궤적,
차량 방향과 경로 오차를 표시합니다.

## 대회용 UDP 전용 주행

대회 측에서 ROS Bridge를 허용하지 않는 경우 아래 **별도 entrypoint**를 사용합니다.
기존 `waypoint_udp_tracking.launch`와 ROS 서비스 기반 코드는 그대로 유지합니다.
새 launch는 SIM과 GPS/Camera/LiDAR/EgoStatus/CmdControl UDP로만 통신하며,
우리 Ubuntu 내부의 노드 연결과 RViz에는 ROS를 사용합니다. rosbridge와
`/Service_MoraiEventCmd`를 실행하거나 조회하지 않습니다. 기존 `field_check`는
ROS 서비스가 필수인 이전 구성의 진단이므로 UDP 전용 합격 판정용으로 쓰지 않습니다.

공식 [MORAI-NetworkModule CmdControl 예제](https://github.com/MORAI-Autonomous/MORAI-NetworkModule/blob/24.R2.0/EgoNetwork/CmdControl/MoraiCmdController.py)의
55-byte EgoCtrlCmd를 사용하며 **ctrl_mode=2(AutoMode), gear=4(D)**를 송신합니다.
기존 송신기의 ctrl_mode=0 유지 및 서비스 mode gate와 달리 Manual 응답이어도
AutoMode를 요청하므로 Q로 미리 전환하지 않습니다. 해당 대회 SIM이 요청을
수락했는지는 `/control/sim_mode`와 `/competition/ego_status`로 확인합니다.

IMU의 `No UDP packet`은 IMU 수신 port 9220에 패킷이 오지 않았다는 뜻입니다.
새 구성은 기본적으로 GPS 위치와 **Ego Vehicle Status UDP의 ENU heading**으로
주행합니다. heading을 quaternion으로 바꾼 `/competition/heading_imu`는 실제 IMU
센서 측정이 아니며 orientation만 제공합니다. `/imu`의 수신 문제를 고친 것처럼
표시하지 않습니다. 기본값에서는 실제 IMU 수신기를 시작하지 않습니다.

### Windows SIM 설정

현재 현장 IP는 Windows `192.168.0.1`, Ubuntu `192.168.0.10`입니다.

| 항목 | 설정 |
| --- | --- |
| Cmd Control | UDP, Windows Host IP `192.168.0.1`, Host 수신 port `9093`, Connect ON |
| Publisher / Ego Vehicle Status (MoraiInfoPublisher) | UDP, Destination IP `192.168.0.10`, Destination port **9092**, Connect ON |
| GPS | UDP, Host port 9130, Destination `192.168.0.10:9230`, Connect ON |
| Camera Front/Left/Right | UDP, Host 9101/9102/9103, Destination `192.168.0.10:9201/9202/9203`, Connect ON |
| LiDAR | UDP, Host 9110, Destination `192.168.0.10:9210`, Connect ON |
| Subscriber / Service | 대회 규정에 따라 UDP 선택; 이 주행은 이들 항목의 명령/응답을 사용하지 않음 |

EgoStatus의 Host port는 SIM 측 설정을 유지하고 **Destination port만 Ubuntu의
`status_port`와 맞춥니다**. 9092가 대회 고정값이라는 뜻은 아닙니다. 현장 지정 포트가
다르면 SIM Destination port와 launch `status_port`를 같은 값으로 바꿉니다.
Cmd Control 역시 실제 Host 수신 port가 다르면 `morai_port`를 바꿉니다.
현재 decoder는 공식 23.R1+의 181-byte 및 24.R2.0의 229-byte 상태 패킷을 지원합니다.

### 실행

기존 주행 launch와 sensor_bridge/field_check launch를 Ctrl+C로 종료합니다.
동일 UDP port 수신기 및 `/ctrl_cmd` 송신기를 중복 실행하지 않습니다. 기존 rosbridge도
종료해 SIM ROS 연결을 사용하지 않습니다. 기존 roscore가 있으면 그대로 유지합니다.

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/unita_ws/devel/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.0.10
export ROS_MASTER_URI=http://localhost:11311

roslaunch unita_launch competition_udp_tracking.launch \
  morai_ip:=192.168.0.1 morai_port:=9093 status_port:=9092 \
  east_offset:=302595 north_offset:=4124145
```

이 offsets는 현재 K-city waypoint용입니다. 다른 map이면 실제 값을 지정합니다.
위 launch 하나가 sensor_bridge, waypoint follower, UDP 제어, localization,
visualizer, RViz를 실행합니다. 별도로 sensor_bridge를 실행할 필요가 없습니다.
기본 속도는 최초 확인용 **10 km/h**, 경로는 무한 반복입니다.
확인 후 `target_speed_kmh:=20` 등으로 변경합니다.

### 확인과 종료

```bash
rostopic echo /control/udp_status
rostopic echo /competition/ego_status
rostopic echo /control/sim_mode
rostopic echo /localization/valid
rostopic hz /competition/heading_imu
rostopic hz /localization/pose
```

정상 상태는 `SENDING`, 실제 UDP mode `2`, localization valid `true`입니다.
`SENDING`은 로컬 송신 성공이며 SIM 수신 ACK는 아닙니다.
`BRAKE: missing/stale/frozen Ego UDP status`이면 Publisher의 Connect, Destination
IP/port를 확인합니다. GPS 또는 heading/IMU가 무효하거나 오래됐거나 command가
오래된 경우에도 속도·조향 대신 full brake를 송신합니다. 같은 timestamp의 패킷을
재송신해도 계속 주행하지 않습니다. SIM timestamp는 Sync Mode에서 다른 clock일 수
있으므로 heading은 수신 시각으로 ROS stamp를 만들고 source timestamp 진행과
wall timeout도 검사합니다.

AutoMode를 반복 요청하므로 실행 중 Q만으로 Manual에 복귀하는 방식은 사용하지
않습니다. 수동 복귀는 launch를 Ctrl+C로 종료합니다. 종료 시 제동과 Keyboard
mode=1을 UDP로 송신합니다. launch를 유지하면서 중지/수동 복귀할 때는
**Ubuntu 내부만 사용하는** 서비스를 이용할 수 있습니다(SIM ROS Service 불필요).

```bash
rosservice call /competition_udp_control/set_enabled "data: false"
# 재개
rosservice call /competition_udp_control/set_enabled "data: true"
```

실제 IMU sensor를 쓰려면 `heading_source:=imu`를 추가합니다. 이때 IMU UDP
Host 9120 → Destination `192.168.0.10:9220`을 Connect ON으로 설정합니다.
EgoStatus UDP는 실제 상태 확인과 freshness 판정을 위해 계속 필요합니다.
차량 모델이 바뀌면 `wheelbase`, `max_steering_deg`를 실제 차량 설정과 맞춥니다.
조향은 기존 부호를 유지합니다. 실제 반전이 확인된 경우에만 `steering_sign:=-1`을
UDP 변환 측에 적용할 수 있습니다. follower 측은 항상 +1로 이중 반전을 피합니다.

## Field Check / 5분 현장점검 절차

현장에서는 Windows MORAI Client PC와 Ubuntu Algorithm PC를 Ethernet LAN으로
연결합니다. 아래 순서로 **통신을 먼저 점검**합니다. `field_check.launch`는 기존
센서 수신기와 진단 노드만 시작합니다. follower, UDP 제어 송신기, RViz는
시작하지 않으며 속도·조향 명령을 발행하거나 모드·기어를 변경하지 않습니다.
이미 실행 중인 주행 노드는 중지하지 않으므로 최초 점검 전 별도로 종료하세요.

### 출발 전 준비

두 workspace가 아래 위치에 있어야 합니다. underlay를 먼저 빌드합니다.

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make
source ~/catkin_ws/devel/setup.bash
cd ~/unita_ws
catkin_make
source ~/unita_ws/devel/setup.bash
catkin_make run_tests
catkin_test_results
rospack find morai_msgs
rospack find rosbridge_server
rospack find velodyne_pointcloud
```

패키지가 없으면 출발 전에 설치/빌드합니다. ROS apt 저장소가 설정된 Noetic
Ubuntu에서 `sudo apt install ros-noetic-rosbridge-server ros-noetic-velodyne`
을 사용할 수 있습니다. `morai_msgs`는 `~/catkin_ws`의 메시지 패키지입니다.
추가 pip 설치는 필요 없습니다.

### 1. LAN 연결과 IP 확인

Windows에서 `ipconfig`, Ubuntu에서 아래 명령으로 **Ethernet** IPv4를 확인합니다.
양쪽 주소와 netmask는 현장 담당자와 정한 같은 subnet에 맞춥니다. Wi-Fi 또는
Tailscale 주소를 센서/제어 주소로 사용하지 않습니다. 이 진단은 네트워크 설정을
변경하지 않습니다.

```bash
ip -4 addr
ip route
```

**모든 Ubuntu 터미널에서** 다음 설정을 실행합니다. 아래 두 IP는 예시이며
반드시 현장에서 확인한 실제 Ethernet IP로 바꾸세요.

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/unita_ws/devel/setup.bash
export UBUNTU_IP=192.168.0.10
export MORAI_IP=192.168.0.20
unset ROS_HOSTNAME
export ROS_IP="$UBUNTU_IP"
export ROS_MASTER_URI=http://localhost:11311
```

`ROS_MASTER_URI`의 localhost는 Ubuntu 자체의 master 주소이므로 정상입니다.
반면 Sensor Destination, ROS Bridge IP, 제어 `morai_ip`의 `127.0.0.1`은
다른 PC에 도달하지 않습니다. 양쪽 PC 사이의 TCP 9090과 아래 센서 UDP 포트를
방화벽에서 허용해야 합니다. ping이 막혀도 센서와 서비스가 통과하면 ICMP만
차단된 것일 수 있습니다.

### 2. ROS와 ROS Bridge 실행

Terminal 1 (이미 master가 있으면 중복 실행하지 않음):

```bash
roscore
```

Terminal 2:

```bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
```

### 3. Windows MORAI Network Settings 확인

- ROS Bridge 연결 IP = **Ubuntu Ethernet IPv4**, port = **9090**, 연결 ON.
- Sensor 프로토콜 = UDP, 각 Destination IP = **Ubuntu Ethernet IPv4**, Connect ON.
- Ego Network의 **Cmd Control** = UDP, 제어 수신 port = **9093**.
  수신 IP를 지정하는 항목이 있으면 **Windows Ethernet IPv4**로 맞춥니다.
- Ego Network의 **Service** = ROS, `/Service_MoraiEventCmd` 연결 ON.
  기본 모드 조회는 ROS 서비스를 사용하므로 Ego Status Publisher를 UDP로 바꿀
  필요가 없습니다. Publisher/Subscriber는 필요한 기존 ROS 설정을 유지합니다.
- Ubuntu 제어 코드의 `morai_ip` = **Windows Ethernet IPv4**.
  Sensor의 Host port와 Ubuntu 수신 Destination port는 서로 다릅니다.

| 센서 | Windows Host port | Ubuntu Destination port | ROS 토픽 |
| --- | ---: | ---: | --- |
| Front camera | 9101 | 9201 | `/camera/image/front` |
| Left camera | 9102 | 9202 | `/camera/image/left` |
| Right camera | 9103 | 9203 | `/camera/image/right` |
| Rear camera (선택) | 9104 | 9204 | `/camera/image/rear` |
| LiDAR | 9110 | 9210 | `/lidar3D` |
| IMU | 9120 | 9220 | `/imu` |
| GPS | 9130 | 9230 | `/gps` |

LiDAR 드라이버 기본값은 **VLP16, 600 RPM**입니다. 실제 대회 SIM의 모델/RPM과
일치하는지 확인하고 필요하면 `lidar_launch`, `lidar_rpm`을 변경합니다.
기본 모델이 대회 센서와 같다고 가정하지 마세요.

### 4. 진단 실행

Terminal 3에서 실행한 뒤 Windows SIM이 센서를 전송하는지 확인합니다.

```bash
roslaunch unita_launch field_check.launch \
  morai_ip:="$MORAI_IP" morai_port:=9093
```

기본 측정 시간은 5초이며 `window:=10`처럼 변경할 수 있습니다. 센서 수신기
시작 직후 실패했다면 SIM Connect를 확인하고, 아래 재점검 명령으로 다시 측정합니다.
이 launch는 출력 완료 후에도 센서 수신기를 유지합니다. 종료하려면 Ctrl+C입니다.

rear는 기본 optional이고 수신기도 시작하지 않습니다. rear까지 필수로 확인하려면
`require_rear_camera:=true`를 추가합니다. rear를 optional 상태로 수신만 하려면
`enable_rear_camera:=true`를 추가합니다. 기존 `sensor_bridge.launch`의 단독 실행은
4개 camera 수신 기본값을 유지합니다.

LiDAR를 제외하는 테스트는 **`enable_lidar:=false require_lidar:=false`**를 둘 다
추가합니다. 수신기만 끄고 필수 조건을 유지하면 LiDAR가 FAIL로 나옵니다.
센서 브릿지가 이미 실행 중이면 `start_sensor_bridge:=false`로 포트 중복 바인딩을
피합니다. ROS Bridge도 함께 시작하려면 별도 Terminal 2 대신 `start_rosbridge:=true`
를 사용합니다. 같은 TCP port의 rosbridge를 중복 실행하지 마세요.

GPS UDP에 map offsets는 포함되지 않습니다. offset 0이어도 GPS 좌표·fix·갱신은
검사하고, offset 문제는 **GPS Map Offset WARN**으로 분리합니다. 실제 map 값을
알면 진단 launch에 다음 인자를 추가합니다.

```bash
# 현재 저장된 K-city waypoint의 값입니다. 다른 대회 map에 그대로 사용하지 마세요.
roslaunch unita_launch field_check.launch \
  morai_ip:="$MORAI_IP" east_offset:=302595 north_offset:=4124145
```

현재 receiver 설정을 유지한 채 다시 진단하거나 종료 코드를 확인할 때:

```bash
rosrun unita_launch field_check.py _morai_ip:="$MORAI_IP"
echo $?
```

map offsets를 설정했다면 위 `rosrun`에도 `_east_offset:=실제숫자`,
`_north_offset:=실제숫자`를 지정합니다. `실제숫자` 문자열 자체를 입력하지 않습니다.
진단 **프로세스**의 종료 코드는 FAIL이 있으면 1, PASS/WARN만 있으면 0입니다.
`roslaunch`는 수신기들을 계속 실행하므로 shell 종료 코드 대신 출력의
`Diagnostic exit code`를 보거나 단독 `rosrun`의 `$?`를 확인하세요.

### 5. 결과 판정과 확인 목록

| 점검 | 통과 기준 |
| --- | --- |
| ROS 환경 | Noetic, master 연결, 모든 필수 패키지 발견 |
| Ethernet | 활성 wired IPv4 존재, Windows IP로 가는 route가 Ethernet 사용 |
| ROS Bridge | 로컬 TCP 9090 listener와 MORAI Event Service 응답 |
| Camera 3개 | Image 타입, 메시지/크기/data 유효, 측정 Hz·frame_id 표시 |
| LiDAR | PointCloud2 타입, 유효 point data 수신, Hz·frame_id 표시 |
| GPS | 유한 좌표, 0/0 아님, status > 0, 갱신·timestamp 정상 |
| IMU | 유한 비영 quaternion, orientation 사용 가능, 갱신·timestamp 정상 |
| 시간 | 수신 지속, stamp가 진행하고 기본 age 1초 이내; stamp 0은 WARN |
| 제어 구성 | `morai_ip`/port 유효, launch 인자 존재, `/ctrl_cmd`가 있으면 CtrlCmd 타입 |

각 항목은 `[PASS]`, `[WARN]`, `[FAIL]`과 가능한 원인을 출력합니다.
수신 Hz는 실제 측정값이며 고정 30/50 Hz 기준으로 합격을 강제하지 않습니다.
1개만 수신하면 Hz 계산은 불가능하다고 표시합니다. WARN이 없으면 `READY`,
WARN만 있으면 `READY WITH WARNINGS`, 필수 항목 FAIL이 있으면 `NOT READY`입니다.
없는 rear와 미설정 offset은 WARN이므로 3-camera 구성의 정상 결과는 보통
`READY WITH WARNINGS`입니다. **이 판정은 통신 진단이며 주행 성능 보장이 아닙니다.**
주행 전에는 map offsets, waypoint/map 좌표 정합, 실제 제어 수신을 별도로 확인합니다.

서비스 검사는 `option=0, gear=-1` 조회만 합니다. 서비스 응답 mode는
1=Manual, 3=ExternalCtrl, 6=Built-in이고 **Manual이어도 서비스 연결은 PASS**입니다.
기어/모드를 설정하지 않습니다. TCP listener만으로 Windows 연결을 증명하지 않으므로
서비스 응답과 센서 수신을 함께 확인합니다. UDP listener와 ROS data 검사도 패킷의
송신 PC를 인증하지 않습니다. 예상치 못한 publisher가 있으면 `rostopic info /gps`와
각 image 토픽을 확인하여 ROS/UDP 브릿지의 중복 센서 발행을 제거합니다.

콘솔 진단 결과와 같은 내용은 `/tmp/unita_field_check_YYYYMMDD_HHMMSS.log`에
저장되고 마지막에 정확한 파일 경로가 표시됩니다. 장애 시 그 파일을 공유하세요.
필수 센서 FAIL이면 출력된 Destination IP/port, Connect, firewall, receiver를
확인합니다. timestamp가 미래/과거이면 ROS clock와 두 PC 시간을 확인합니다.
기본 센서 브릿지는 Ubuntu 수신 시각으로 stamp를 찍으므로 stamp가 정상이어도
SIM 내부 timestamp 동기화까지 확인한 것은 아닙니다.

통신 확인 후 실제 주행은 **별도 명령**입니다. 진단의 receiver에 해당 map의
실제 offsets가 적용된 것을 확인하고 실행합니다. offset을 나중에 바꾸면 Terminal 3의
진단 launch를 Ctrl+C로 종료하고 올바른 offsets로 재실행합니다.

```bash
roslaunch unita_launch waypoint_udp_tracking.launch \
  morai_ip:="$MORAI_IP" morai_port:=9093
```

이미 주행 중인 노드를 읽기만 하는 추가 진단:

```bash
roslaunch unita_launch field_check.launch \
  morai_ip:="$MORAI_IP" start_sensor_bridge:=false monitor_control:=true
```

이 옵션은 `/ctrl_cmd`, `/control/sim_mode`, `/control/udp_status`를 추가 구독하고
현재 sender의 IP/port를 비교합니다. `/control/sim_mode`는 service 응답 숫자와
달리 내부 정규화 값 1=Manual, 2=ExternalCtrl, 0=Built-in/기타/unknown입니다.
`SENDING`은 Ubuntu 송신 성공만 뜻하며 Windows 수신/차량 적용 ACK가 아닙니다.
watchdog의 `BRAKE` 상태는 WARN으로 표시됩니다. 진단 자체는 패킷을 보내지 않습니다.

현장 진단 파일: `scripts/field_check.py`, `src/unita_visualization/field_diagnostics.py`,
`launch/field_check.launch`, `config/field_check.yaml`, `test/test_field_diagnostics.py`.
기존 driving algorithm과 sensor/control UDP protocol은 변경하지 않았습니다.

## I 키로 실제 차량을 웨이포인트 시작점에 초기화

`config/waypoint_start.json`은 MORAI의 **MapInitSetting** 형식입니다.
웨이포인트의 첫 XYZ와 첫 진행 방향으로 초기 스폰 위치·자세를 지정합니다.
더미 차량 배치나 차량을 수동으로 시작점까지 이동하는 단계는 필요 없습니다.

현재 설정값:

- ENU x: `-131.68979755061446`
- ENU y: `-428.3310229377821`
- ENU z: `28.543960281954277`
- ENU yaw: `61.29811514697516°`

현재 설치 SIM의 `MapInitSetting` 클래스에 맞게 JSON 키는 `m_InitPos`,
`m_InitRot`이며 두 값은 x/y/z 벡터입니다. 회전 벡터는 ENU roll/pitch/yaw(도)이고
SIM이 내부 Unity 좌표계로 변환합니다. 초기 위치를 불러올 때 SIM이 노면 높이와
차량 wheel radius를 적용합니다.

로컬 샘플 시나리오의 mapInfo에서 확인한 `R_KR_PR_K-city_2025` 맵에 사용할
파일을 아래 경로에 준비했습니다.

```text
~/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/MapInitSetting/R_KR_PR_K-city_2025/waypoint_start.json
```

SIM에서 **Edit → Map Init Settings → Load Init Ego State**를 열고
`waypoint_start.json`을 선택한 뒤 **Apply**합니다. 이미 창이 열려 있었다면
닫고 다시 열어 목록을 새로 읽습니다. 이제 SIM 화면에서 I를 누르면 **실제 Ego
차량**이 저장된 시작 위치로 초기화됩니다. 파일 생성만으로 실행 중인 SIM의
설정이 자동 적용되지는 않으므로 **한 번 Load/Apply하는 단계가 필요**합니다.

다른 맵이라 목록에 파일이 보이지 않으면 Load 창의 **Open Folder**로 실제
맵의 초기 설정 폴더를 열고 `config/waypoint_start.json`을 그곳에 복사합니다.
이 좌표는 현재 waypoint 파일과 해당 지도 기준입니다.

웨이포인트 시작 위치가 바뀌면 생성 도구로 새 파일을 준비할 수 있습니다.
도구는 파일만 만들며 차량 이동이나 제어 패킷 송신을 수행하지 않습니다.
기존 파일은 덮어쓰지 않으므로 새 파일 이름을 지정합니다.

```bash
rosrun unita_launch generate_start_spawn.py \
  --waypoint-file "$(rospack find unita_waypoint)/config/waypoints.csv" \
  --output /tmp/waypoint_start_new.json
```

주행 중 위치가 한 번에 10 m를 초과해 바뀌면 follower는 전체 경로를 다시
탐색하고 RViz는 이전 궤적·오차 통계를 지웁니다. 10 m 이하의 초기화까지 진행
상태를 확실히 초기화하려면 waypoint 주행 launch를 재실행합니다.
외부 제어에서 I 입력이 허용되는지 및 생성 설정의 실제 적용은 SIM에서 확인해야
합니다. I가 막히면 수동 모드에서 초기화한 뒤 외부 제어로 복귀합니다.

참고: [MORAI 초기 상태 불러오기/적용](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/26.R1/ego-vehicle).

## 카메라 4개 보기

sensor_bridge를 실행한 다음 별도 터미널에서 카메라 전용 RViz를 엽니다.
이 launch는 RViz만 시작하며 센서 수신기나 차량 제어는 시작하지 않습니다.

```bash
source ~/unita_ws/devel/setup.bash
roslaunch unita_launch camera_visualization.launch
```

Front/Left/Right/Rear camera의 Image display는 각각
`/camera/image/front`, `/camera/image/left`, `/camera/image/right`,
`/camera/image/rear`를 raw transport로 표시합니다. 네 번째 카메라는 후방으로
가정했습니다. MORAI Host 9104 → ROS Destination 9204 설정이 필요합니다.
기존 주행 RViz 구성에도 네 개의 Image display가 포함되어 있습니다.

## UDP 차량 제어로 주행

모든 waypoint 주행 launch의 `loop_path` 기본값은 `true`입니다. 마지막 지점에서
첫 지점으로 경로 탐색을 이어서 반복 주행하며, 시작/끝 좌표가 겹쳐도 도착으로
정지하지 않습니다. 단일 주행 후 도착 정지는 `loop_path:=false`로 선택합니다.

MORAI Cmd Control이 UDP일 때는 `waypoint_udp_tracking.launch`를 사용합니다.
이 launch는 아래 기존 주행/시각화 구성에 `control/morai_udp_control` 송신 노드를
추가합니다. `MORAI_IP`는 실제 SIM PC의 IPv4, 9093은 SIM의 제어 수신 포트와
일치시켜야 합니다. sensor_bridge는 별도 실행하며 지도 offsets를 설정합니다.
기본 모드 확인은 기존 ROS `/Service_MoraiEventCmd`를 `option=0`으로 조회합니다.
Service는 ROS로 유지하세요. Ego Vehicle Status Publisher를 UDP로 바꾸거나
9092로 상태를 전송할 필요가 없습니다. 수동 또는 내장 자율주행에서는 송신을
멈추고, 외부 제어에서는 새 waypoint 명령으로 주행을 재개합니다.

```bash
roslaunch unita_launch waypoint_udp_tracking.launch \
  morai_ip:="$MORAI_IP" morai_port:=9093
```

기존 주행/시각화가 이미 켜져 있으면 UDP 송신 노드만 추가하세요.

```bash
roslaunch control morai_udp_control.launch morai_ip:="$MORAI_IP" morai_port:=9093
```

최대 차량 조향각과 MORAI 설정·송신 상태 확인은 `control/README.md`에 설명합니다.

## 실행

```bash
source ~/catkin_ws/devel/setup.bash
cd ~/unita_ws
catkin_make
source devel/setup.bash
```

첫 번째 터미널에서 MORAI SIM의 UDP 전송을 켜고 sensor_bridge를 실행합니다.
GPS 지도 offsets를 사용 중인 지도의 실제 값으로 지정해야 합니다.

```bash
# 아래 변수에는 실제 지도의 offset 숫자를 지정한 후 실행합니다.
roslaunch sensor_bridge sensor_bridge.launch \
  east_offset:="$EAST_OFFSET" north_offset:="$NORTH_OFFSET"
```

새 터미널에서 주행과 시각화를 함께 시작합니다.

```bash
source ~/unita_ws/devel/setup.bash
roslaunch unita_launch waypoint_tracking.launch
```

이 launch는 **기존 주행 노드도 시작**합니다. 기본값은 `local_xy`,
기존 `waypoints.csv`, 목표 속도 20 km/h이며 `/ctrl_cmd`로 제어 명령을 발행합니다.
SIM에 실제 적용하려면 기존 MORAI 차량 제어 연결이 필요합니다.
sensor_bridge는 센서 수신 전용이며 차량 제어 UDP 송신을 구현하지 않습니다.

다른 waypoint 파일을 사용하려면:

```bash
roslaunch unita_launch waypoint_tracking.launch \
  waypoint_file:=/absolute/path/waypoints.csv waypoint_format:=local_xy
```

**기존 `gps_waypoint.launch`가 이미 실행 중이면** 중복 실행하지 말고 시각화만 추가합니다.

```bash
roslaunch unita_launch waypoint_visualization.launch
```

localization도 이미 실행 중이면 `start_localization:=false`를 추가합니다.
GUI를 다른 PC에서 실행할 경우 `start_rviz:=false`로 노드만 실행할 수 있습니다.
이때 두 PC가 같은 ROS master와 올바른 ROS_IP/ROS_HOSTNAME을 사용해야 합니다.

RViz를 나중에 별도로 열려면:

```bash
rviz -d $(rospack find unita_launch)/config/waypoint_tracking.rviz
```

RViz 미설치 시 `sudo apt install ros-noetic-rviz`로 설치합니다.

## 화면에서 보는 것

Fixed Frame은 **map**입니다. 모든 표시가 map 좌표를 사용하므로 TF 발행은
필수가 아닙니다. 기본 화면은 현재 저장된 waypoint의 시작점 주변을 위에서 봅니다.
다른 경로를 사용할 경우 Views의 Current View X/Y를 현재 위치에 맞추고,
마우스 드래그/휠로 이동·확대하거나 Focus Camera 도구로 차량을 클릭합니다.

| 표시 | 의미 |
| --- | --- |
| 하늘색 선 | 실제 follower가 발행한 기준 경로 `/waypoint_path` |
| 주황색 선 | GPS+IMU localization에서 수집한 실제 궤적 |
| 초록 화살표 | 현재 위치 및 IMU yaw, 길이는 3 m로 고정 (속도 아님) |
| 자홍색 점·선 | 경로 선분의 최근접 지점과 차량까지의 거리 |
| 파란 구 | 경로 시작점 |
| 빨간 구 | 경로 마지막 점 (종료 판정 자체를 뜻하지 않음) |
| 흰색 글자 | 현재 오차, 최근 RMS/최대 오차, 경로 진행 거리, 종점까지 거리 |

최근접 점은 **시각화용 투영점**입니다. Pure Pursuit가 실제로 선택한 lookahead
목표점이나 waypoint index를 표시하는 것은 아닙니다. Segment는 시각화가 선택한
선분의 0부터 시작하는 index입니다.

| 수치 | 단위와 해석 |
| --- | --- |
| CTE | 가장 가까운 경로 선분까지의 거리(m), 경로 진행 방향 왼쪽은 +, 오른쪽은 − |
| Heading | IMU yaw − 최근접 선분 방향(deg), −180~180으로 정규화 |
| Window RMS / Max | 최근 최대 200개 서로 다른 pose timestamp의 CTE RMS / 절댓값 최대 |
| Along | 최근접 지점까지 기준 경로를 따라 잰 거리(m), 종점까지 남은 거리가 아님 |
| Goal | 마지막 waypoint까지 직선 거리(m) |

선분 끝 바깥에서는 CTE에 종방향 거리도 포함됩니다. 교차로·겹친 경로에서는
전역 최근접 선분이 바뀌어 Along/Heading이 갑자기 변할 수 있습니다.
곡선이나 모서리에서는 선분 방향과 lookahead 방향이 달라 Heading이 커질 수 있습니다.

## 분석 순서

1. **좌표 정합부터 확인**합니다. 경로와 차량이 수십~수백 km 떨어지면 waypoint
   좌표 형식, UTM zone, eastOffset/northOffset을 먼저 확인합니다. 현재 저장된
   waypoint는 local_xy이고 시작점이 대략 x=−132, y=−428 m입니다.
2. **직선에서 궤적이 하늘색 선을 따라가는지** 확인합니다. CTE가 0에 가까워지고
   화살표가 진행 방향과 일치하는지 봅니다. 꾸준한 한쪽 오차는 지도 정합,
   센서 장착 위치, yaw 기준 또는 제어 편향을 점검할 근거입니다.
3. **곡선에서 바깥으로 벌어지거나 안쪽을 잘라 가는지** 확인합니다. 오차 증가가
   특정 곡선에서 반복되는지, 속도와 lookahead를 변경했을 때 어떻게 달라지는지
   비교합니다. 시각화 자체는 제어 파라미터를 변경하지 않습니다.
4. **좌우 진동이 지속되는지** 확인합니다. CTE 부호가 반복해서 바뀌고 Heading도
   진동하면 lookahead, 속도, 조향 부호/한계 등을 함께 점검합니다.
5. **종점에서 정지하는지** 실제 차량과 명령을 함께 확인합니다. Goal이 작다고
   종료가 보장되지는 않습니다. 기존 follower는 waypoint index와 goal_tolerance도
   사용하며 기본 허용 거리는 2 m입니다.

오차 허용 기준은 도로 폭, 차량 폭, 요구 정확도로 정해야 합니다. 예를 들어
0.5 m 기준을 쓰더라도 평균만 보지 말고 직선/곡선의 최대 오차와 반복성을 함께
비교하세요. 이 측정은 GPS 기반 localization과 기준 경로 간의 비교입니다.
GPS 자체의 오차나 공통 좌표 보정 오류를 검출하는 독립 ground truth는 아닙니다.

센서가 무효하거나 오래되면 차량 marker와 오차 발행이 중지됩니다. 마지막 궤적은
남아 있을 수 있으나 유효 위치를 뜻하지 않습니다. 복구 시 궤적과 RMS 창을 초기화해
GPS 단절 구간을 직선으로 연결하지 않습니다. 궤적은 최대 10000점, 기본 0.2 m
간격으로 보관합니다. 시각화의 invalid 처리가 기존 follower의 주행을 중지시키는
것은 아닙니다. 기존 follower의 센서 손실 처리는 아직 별도 개선 전입니다.

## 토픽 확인 및 기록

```bash
rostopic echo /waypoint_debug/status
rostopic echo /waypoint_debug/cross_track_error
rostopic echo /waypoint_debug/heading_error_deg
rostopic echo /localization/valid
```

오차 값은 상태가 유효할 때만 갱신됩니다. 수치만 확인하지 말고 status/valid와
수신 시각도 함께 확인하세요.

주행 비교를 위해 bag으로 저장할 수 있습니다. 기준 경로가 latched 토픽이므로
기록 시작 후에도 한 번 수신됩니다. 저장 파일명/경로는 원하는 위치로 지정하세요.

```bash
rosbag record -O waypoint_tracking.bag \
  /gps /imu /waypoint_path /localization/pose /localization/valid /ctrl_cmd \
  /waypoint_debug/cross_track_error /waypoint_debug/heading_error_deg \
  /waypoint_debug/goal_distance /waypoint_debug/progress_m /waypoint_debug/status
```

## 파일과 검증

- `scripts/waypoint_visualizer.py`: 궤적, MarkerArray, 상태 및 오차 토픽 발행
- `src/unita_visualization/path_metrics.py`: 최근접 선분 투영과 오차 계산
- `launch/waypoint_tracking.launch`: 기존 주행 + localization + 시각화 + RViz
- `launch/waypoint_visualization.launch`: localization + 시각화 + RViz (제어 노드 미시작)
- `config/waypoint_tracking.rviz`: map 기준 상부 시점과 색상 설정
- `test/test_path_metrics.py`: 부호·투영·방향 wrapping·중복점·곡선 모서리 검증

빌드와 수학 테스트, 별도 ROS master의 합성 데이터 통합 테스트로 토픽/marker 발행,
센서 무효·timeout·복구·경로 변경을 검증합니다. 실제 SIM에서 추종 성능 및 RViz GUI
표시는 아래 실행 방법으로 확인해야 합니다.

```bash
catkin_make run_tests_unita_launch
catkin_test_results build/test_results/unita_launch
```
