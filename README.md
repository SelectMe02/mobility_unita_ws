
# UNITA - 2026 창작모빌리티 경진대회 자율주행 시스템

인천대학교 자율주행 동아리 **UNITA**의 2026 창작모빌리티 경진대회 참가팀 **유니타스타(UNITASTAR)**에서 개발한 MORAI 시뮬레이터 기반 자율주행 워크스페이스입니다.

본 프로젝트는 **ROS1 Noetic** 환경에서 개발되었으며, MORAI 시뮬레이터와 UDP 통신을 통해 센서 데이터를 수신하고 차량을 제어합니다.

GPS/IMU 기반 위치추정, LiDAR 장애물 인식, YOLO 기반 신호등 인식, Waypoint 및 Raceline 경로 추종, 동적 장애물 회피, GPS 음영 구간 주행 등을 하나의 통합 시스템으로 구현하였습니다.

최종 대회 주행에는 `unita_launch/launch/extreme.launch`를 사용하였습니다.

---

## 1. 대회 및 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 대회 | 2026 창작모빌리티 경진대회 |
| 참가팀 | 유니타스타(UNITASTAR) |
| 소속 | 인천대학교 자율주행 동아리 UNITA |
| 시뮬레이터 | MORAI SIM |
| 운영체제 | Ubuntu / ROS1 Noetic |
| 통신 | UDP 기반 센서 수신 및 차량 제어 |
| 주요 센서 | Camera, 3D LiDAR, GPS, IMU |
| 최종 주행 Launch | `extreme.launch` |
| 경로 데이터 | `map_extreme_project.json` |

본 프로젝트는 대회의 여러 주행 환경에 대응하기 위해, 구간에 따라 주행 알고리즘을 전환하는 통합 자율주행 구조를 사용합니다.

### 주요 구현 기능

- **UDP Sensor Bridge**: MORAI의 Camera, GPS, IMU, 3D LiDAR 데이터를 ROS 토픽으로 변환
- **Localization**: GPS/IMU 기반 차량 위치 및 방향 추정
- **Waypoint Tracking**: 저장된 경로점과 목표 속도를 활용한 주행
- **Raceline Following**: 복수의 주행 경로를 이용한 경로 추종 및 차선 선택
- **Obstacle Avoidance**: LiDAR 기반 전방·후방 장애물 인식과 회피
- **Traffic Light Recognition**: YOLO 기반 신호등 인식 및 정지 판단
- **GPS Blackout Navigation**: GPS 음영 구간에서 LiDAR Odometry와 IMU를 활용한 주행
- **Control Arbitration**: 주행 구간에 따라 제어 명령을 선택하고 통합
- **Map Tuner / Map Debug**: 주행 경로 편집, 속도 설정 및 실시간 차량 상태 시각화

---

## 2. 워크스페이스 구성

본 프로젝트는 기능별 ROS 패키지로 구성되어 있습니다.

```text
unita_ws/
├── src/
│   ├── sensor_bridge/
│   │   ├── scripts/
│   │   ├── launch/
│   │   └── config/
│   │
│   ├── localization/
│   │   ├── scripts/
│   │   └── config/
│   │
│   ├── control/
│   │   ├── scripts/
│   │   └── config/
│   │
│   ├── perception/
│   │   └── scripts/
│   │
│   ├── unita_waypoint/
│   │   ├── scripts/
│   │   ├── launch/
│   │   └── config/
│   │
│   ├── leo_avoidance/
│   │   ├── scripts/
│   │   └── config/
│   │
│   ├── leo_guard/
│   │   ├── scripts/
│   │   └── config/
│   │
│   └── unita_launch/
│       ├── scripts/
│       ├── launch/
│       │   └── extreme.launch
│       └── config/
│
└── best1.pt
```

### ROS 패키지별 역할

| 패키지 | 주요 기능 |
|---|---|
| `sensor_bridge` | MORAI UDP 센서 데이터 수신 및 ROS 토픽 변환 |
| `localization` | GPS/IMU 기반 차량 위치추정 |
| `control` | UDP 제어 관련 설정 및 제어 기능 |
| `perception` | YOLO 기반 신호등 객체 인식 |
| `unita_waypoint` | Waypoint 추종, Raceline 주행, LiDAR 장애물 인식 및 Map Tuner |
| `leo_avoidance` | 동적 장애물 회피, GPS 음영 구간 주행 및 UDP 제어 송신 |
| `leo_guard` | 신호등 판단, 정지 조건 처리 및 제어 명령 선택 |
| `unita_launch` | 전체 주행 시스템 실행 및 시각화 |

---

## 3. 개발 환경 및 설치

### 3.1 개발 환경

- Ubuntu (ROS1 Noetic 지원 환경)
- ROS1 Noetic
- Python 3
- Catkin Build System
- MORAI SIM
- RViz
- OpenCV
- PyTorch / Ultralytics YOLO
- Velodyne ROS Driver

ROS Noetic은 기본적으로 Ubuntu 20.04를 대상으로 합니다.

필요한 ROS 패키지와 Python 라이브러리는 각 패키지의 `package.xml`, 소스코드 및 실행 환경에 맞게 설치해야 합니다.

### 3.2 GitHub 저장소 다운로드

MORAI 관련 메시지 패키지를 포함한 `catkin_ws`와 자율주행 코드가 포함된 `unita_ws`를 각각 다운로드합니다.

```bash
git clone https://github.com/SelectMe02/catkin_ws.git ~/catkin_ws

git clone https://github.com/SelectMe02/mobility_unita_ws.git ~/unita_ws
```

### 3.3 워크스페이스 빌드

먼저 MORAI 메시지 관련 워크스페이스를 빌드합니다.

```bash
source /opt/ros/noetic/setup.bash

cd ~/catkin_ws
catkin_make

source ~/catkin_ws/devel/setup.bash
```

이후 자율주행 워크스페이스를 Overlay 형태로 빌드합니다.

```bash
cd ~/unita_ws
catkin_make

source ~/unita_ws/devel/setup.bash
```

새 터미널을 사용할 때는 다음 환경 설정을 적용합니다.

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
source ~/unita_ws/devel/setup.bash
```

---

## 4. 전체 자율주행 시스템

### 4.1 최종 주행 파일

대회 최종 주행에는 다음 Launch 파일을 사용합니다.

```text
src/unita_launch/launch/extreme.launch
```

해당 Launch 파일은 센서 수신, 위치추정, 주행 경로 추종, 장애물 회피, 신호등 인식, 차량 제어를 함께 실행합니다.

### 4.2 전체 동작 구조

```text
                  MORAI Simulator
                         |
                  UDP Sensor Data
                         |
                  sensor_bridge
                         |
          +--------------+--------------+
          |              |              |
       GPS / IMU       3D LiDAR       Camera
          |              |              |
     Localization   Obstacle Detect   YOLO
          |              |              |
          +-------+------+              |
                  |                     |
          Driving State                 |
                  |                     |
       +----------+----------+          |
       |                     |          |
 Raceline Follower     Leo Follower     |
       |                     |          |
       |              Route Avoidance   |
       |                     |          |
       +----------+----------+----------+
                  |
             Signal Guard
                  |
              /ctrl_cmd
                  |
         UDP Control Sender
                  |
             MORAI Vehicle
```

**주요 특징**

- 하나의 Launch에서 전체 주행 기능을 실행합니다.
- 주행 구간에 따라 Raceline Follower와 Leo Follower의 제어 명령을 선택합니다.
- 신호등 판단 및 주행 제어 통합은 Signal Guard에서 수행합니다.
- `/ctrl_cmd` 최종 명령은 UDP 제어 노드를 통해 MORAI 차량에 전달됩니다.

### 4.3 주행 구간별 알고리즘

| 구간 | 주행 방식 |
|---|---|
| 출발 ~ CP7 | Leo Waypoint Follower + 장애물 회피 + 신호등 판단 |
| CP7 ~ CP11 | Raceline Follower 기반 경로 추종 및 장애물 대응 |
| CP11 ~ 출발 지점 | Leo Waypoint Follower 기반 주행 |
| GPS 음영 터널 구간 | LiDAR Odometry + IMU + Follow The Gap(FTG) |

#### Raceline Follower

CP7~CP11 구간에서는 저장된 Raceline을 기반으로 주행합니다.

- 여러 Raceline 중 주행 가능한 경로를 선택
- LiDAR 기반 전방 및 후방 장애물 확인
- 차량 간 거리와 주행 조건을 고려한 차선 변경
- 목표 속도 기반 종방향 PID 제어
- Sector 11에서 Raceline 3 우선 주행 설정

#### Leo Waypoint Follower

그 외 구간에서는 Leo Follower를 사용합니다.

- GPS 기반 Waypoint 추종
- 장애물 위치를 고려한 회피 목표 생성
- 차선 변경 가능 구간을 고려한 경로 선택
- 신호등 및 정지 조건 반영
- GPS 음영 구간 진입 시 주행 위치 정보 전환

#### GPS 음영 구간

터널 등 GPS 데이터가 제한되는 구간에서는 LiDAR Odometry와 IMU 기반 주행 상태를 활용합니다.

- GPS 기반 위치추정과 LiDAR 기반 주행 위치 정보 전환
- IMU Heading 활용
- LiDAR 기반 Follow The Gap 장애물 회피
- GPS 복구 후 정상 경로 주행으로 전환

#### Signal Guard

Signal Guard는 두 Follower의 주행 명령을 통합합니다.

- CP7~CP11: Raceline Follower 제어 명령 사용
- 그 외 구간: Leo Follower 제어 명령 사용
- 신호등 인식 결과와 정지 조건 처리
- 최종 `/ctrl_cmd` 발행

---

## 5. MORAI UDP 통신 설정

본 프로젝트는 대회용 UDP 통신을 통해 센서 데이터를 수신하고 차량 제어 명령을 송신합니다.

MORAI 시뮬레이터의 센서 Destination IP 및 Port와 다음 파일의 설정을 일치시켜야 합니다.

```text
src/sensor_bridge/config/sensors.yaml
```

### 5.1 네트워크 설정

MORAI 시뮬레이터와 ROS가 서로 다른 PC에서 실행되는 경우:

- **MORAI PC**: 시뮬레이터 실행 및 센서 데이터 송신
- **Ubuntu PC**: ROS 센서 수신, 자율주행 알고리즘 실행 및 제어 명령 송신

센서 데이터의 Destination IP는 Ubuntu PC의 수신 IP로 설정합니다.

`extreme.launch`의 `morai_ip`는 MORAI 시뮬레이터가 실행되는 PC의 IP로 지정합니다.

### 5.2 기본 네트워크 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `morai_ip` | `192.168.0.1` | MORAI 시뮬레이터 PC IP |
| `morai_port` | `9093` | 차량 제어 UDP 포트 |
| `status_port` | `9096` | 차량 상태 관련 UDP 포트 |
| `east_offset` | `302595` | 지도 East Offset |
| `north_offset` | `4124145` | 지도 North Offset |

위 Offset은 현재 K-City 주행 경로 기준으로 설정되어 있습니다.

다른 지도를 사용하는 경우 해당 지도에 맞는 Offset으로 변경해야 합니다.

---

## 6. 최종 자율주행 실행

### 6.1 기본 실행

MORAI 시뮬레이터를 실행하고 UDP 통신 설정을 완료한 후 Ubuntu에서 다음 명령어를 실행합니다.

```bash
source ~/catkin_ws/devel/setup.bash
source ~/unita_ws/devel/setup.bash

roslaunch unita_launch extreme.launch
```

기본 설정은 다음과 같습니다.

```text
MORAI IP       : 192.168.0.1
Control Port   : 9093
Status Port    : 9096
RViz           : Disabled
Diagnostic Log : Disabled
```

`extreme.launch`는 센서 브릿지와 주행 노드를 함께 실행하므로 `sensor_bridge.launch`를 별도로 중복 실행할 필요가 없습니다.

### 6.2 MORAI IP 지정 실행

시뮬레이터 IP가 다른 경우 실행 인자로 변경할 수 있습니다.

```bash
roslaunch unita_launch extreme.launch \
  morai_ip:=192.168.0.10 \
  morai_port:=9093 \
  status_port:=9096
```

위 IP는 예시이며 실제 MORAI PC의 IP로 변경해야 합니다.

### 6.3 RViz 포함 실행

```bash
roslaunch unita_launch extreme.launch \
  start_rviz:=true
```

이 옵션은 `waypoint_tracking.rviz` 설정으로 RViz를 실행합니다.

상세 주행 경로와 차량 상태를 확인하려면 별도의 `map_debug.launch`를 사용하는 것을 권장합니다.

### 6.4 주행 로그 활성화

```bash
roslaunch unita_launch extreme.launch \
  log:=true
```

로그를 활성화하면 주행 상태 및 진단 데이터 기록 노드가 추가 실행됩니다.

기본 기록 경로는 다음과 같습니다.

```text
~/unita_ws/src/unita_waypoint/log/
```

### 6.5 주요 실행 옵션

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `morai_ip` | `192.168.0.1` | MORAI 시뮬레이터 IP |
| `morai_port` | `9093` | UDP 제어 포트 |
| `status_port` | `9096` | 차량 상태 포트 |
| `waypoint_file` | `map_extreme_project.json` | 주행 경로 JSON |
| `model_path` | `best1.pt` | YOLO 모델 가중치 |
| `raceline_checkpoints` | `7-11` | Raceline Follower 적용 구간 |
| `waypoint_only_point_ids` | `3496-3572` | Waypoint 중심 주행 구간 |
| `r1_only_point_ids` | `708-903` | Raceline 1 유지 구간 |
| `highway_sectors` | `11` | 고속도로 주행 로직 적용 Sector |
| `sector11_min_speed_kmh` | `30.0` | Sector 11 속도 하한 설정 |
| `simulator_s11_no_stop` | `true` | S11 정지 요청에 대한 시뮬레이터 전용 속도 처리 |
| `max_steering_deg` | `29.0` | UDP 조향 정규화 기준 |
| `start_rviz` | `false` | RViz 자동 실행 여부 |
| `log` | `false` | 진단 로그 기록 여부 |

`simulator_s11_no_stop`는 Sector 11의 정지 요청을 최소 속도 목표로 변경하는 대회 시뮬레이터 전용 설정이므로 실제 차량에서 그대로 사용해서는 안 됩니다.

---

## 7. Map Debug - 주행 시각화 및 디버깅

Map Debug는 K-City 지도 위에 차량 위치와 주행 경로를 표시하고, 자율주행 상태를 확인하기 위한 시각화 도구입니다.

### 7.1 실행 방법

먼저 주행 시스템을 실행합니다.

**Terminal 1 - 자율주행 실행**

```bash
source ~/unita_ws/devel/setup.bash

roslaunch unita_launch extreme.launch
```

**Terminal 2 - Map Debug 실행**

```bash
source ~/unita_ws/devel/setup.bash

roslaunch unita_waypoint map_debug.launch
```

Map Debug는 기본적으로 RViz를 함께 실행합니다.

### 7.2 주요 기능

- K-City 지도 및 Raceline 시각화
- 실시간 차량 위치 및 Heading 표시
- 현재 Sector 및 Checkpoint 확인
- 현재 차량 속도 표시
- 주행 목표 경로 확인
- 장애물 인식 결과 및 주변 주행 환경 확인
- 차선 선택 및 Route Avoidance 상태 확인
- Localization 유효 상태 확인

Map Debug는 주행 제어를 수행하는 노드가 아닌 시각화 및 진단 도구입니다.

### 7.3 주요 ROS 토픽 확인

ROS 토픽 수신 여부는 다음 명령어로 확인할 수 있습니다.

```bash
rostopic list
```

위치추정:

```bash
rostopic echo /localization/pose
```

차량 속도:

```bash
rostopic echo /competition/ego_speed_kmh
```

최종 차량 제어 명령:

```bash
rostopic echo /ctrl_cmd
```

Leo Follower 주행 상태:

```bash
rostopic echo /leo/waypoint_debug/tracking_state
```

Route Avoidance:

```bash
rostopic echo /leo/route_avoidance
```

최종 제어 명령뿐 아니라 각 Follower의 상태를 함께 확인하면 주행 구간 전환 및 제어 동작을 분석할 수 있습니다.

### 7.4 주행 지도 파일 주의사항

최종 주행에 사용한 경로 파일은 다음과 같습니다.

```text
src/unita_waypoint/config/map_extreme_project.json
```

반면 Map Debug의 기본 프로젝트 파일은 다음과 같습니다.

```text
src/unita_waypoint/config/map_tunner_project.json
```

따라서 두 JSON 파일이 서로 다르면 실제 주행 경로와 시각화된 경로에 차이가 발생할 수 있습니다.

Map Debug가 최종 주행 경로와 동일한 데이터를 표시하도록 하려면, 관련 코드의 경로 처리 방식도 확인한 후 동일한 프로젝트 파일을 사용해야 합니다.

---

## 8. Map Tuner - 주행 경로 및 속도 편집

Map Tuner는 K-City 지도에서 Waypoint와 Raceline을 편집하고 구간별 목표 속도를 지정하기 위해 개발한 RViz 기반 도구입니다.

### 8.1 실행 방법

```bash
source ~/unita_ws/devel/setup.bash

roslaunch unita_waypoint map_tunner.launch
```

ROS Master가 실행 중이지 않다면 먼저 다른 터미널에서 다음 명령어를 실행합니다.

```bash
roscore
```

Map Tuner는 MORAI 시뮬레이터와 연결하지 않고도 경로를 편집할 수 있습니다.

### 8.2 주요 기능

- K-City 지도 표시 및 위치 보정
- Sector별 주행 경로 관리
- Raceline 1, 2, 3 생성 및 편집
- Waypoint 추가, 삭제 및 위치 변경
- 개별 Waypoint 목표 속도 설정
- 선택한 구간의 목표 속도 일괄 설정
- 주행 경로 JSON 저장

### 8.3 구간 구성

K-City 주행 경로는 START 및 Checkpoint를 기준으로 Sector를 구분합니다.

| 구간 | 설명 |
|---|---|
| S1 | START → CP1 |
| S2 | CP1 → CP2 |
| S3 ~ S14 | 각 Checkpoint 사이 구간 |
| S15 | CP14 → START |

각 Sector에서는 Raceline 1~3을 독립적으로 관리할 수 있습니다.

### 8.4 속도 설정

Map Tuner에서는 각 Waypoint의 목표 속도를 `speed_kmh`로 지정합니다.

```json
{
  "id": 1,
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "speed_kmh": 30.0
}
```

위 코드는 데이터 구조를 설명하기 위한 예시이며 실제 저장된 좌표와는 다릅니다.

최종 `extreme.launch`의 Follower들은 프로젝트 JSON에 저장된 목표 속도를 활용합니다. 실제 제어 속도는 장애물, 곡률, 안전 조건 및 제어 로직에 의해 조정될 수 있습니다.

### 8.5 프로젝트 파일 저장

Map Tuner의 기본 프로젝트 파일은 다음과 같습니다.

```text
src/unita_waypoint/config/map_tunner_project.json
```

반면 최종 대회 주행용 파일은 다음과 같습니다.

```text
src/unita_waypoint/config/map_extreme_project.json
```

최종 주행용 프로젝트를 직접 편집하려면 다음과 같이 실행할 수 있습니다.

```bash
roslaunch unita_waypoint map_tunner.launch \
  project_file:=$(rospack find unita_waypoint)/config/map_extreme_project.json
```

**주의:** 위 명령어는 최종 주행 경로 파일을 직접 편집 대상으로 사용하므로 실행 전에 백업하는 것을 권장합니다.

경로를 수정하고 JSON을 저장한 경우, 변경된 경로를 적용하려면 주행 Launch를 다시 실행해야 할 수 있습니다.

자세한 편집 방법은 다음 문서를 참고합니다.

[MAP_TUNNER.md](src/unita_waypoint/MAP_TUNNER.md)

---

## 9. 주행 데이터 기록 및 디버깅

### 9.1 전체 주행 로그 활성화

```bash
roslaunch unita_launch extreme.launch \
  log:=true
```

해당 옵션은 다음과 같은 정보를 기록하기 위한 노드를 활성화합니다.

- Raceline Follower 주행 판단
- 장애물 인식 및 회피 상태
- Leo Follower 주행 상태
- 신호등 및 Signal Guard 상태
- GPS 음영 구간의 주행 상태
- 차량 제어 명령
- 차량 속도 및 위치 정보

### 9.2 ROS Bag 기록

필요한 토픽을 별도로 저장할 수도 있습니다.

```bash
rosbag record -O extreme_driving.bag \
  /gps \
  /imu \
  /lidar3D \
  /localization/pose \
  /leo/driving_pose \
  /leo/driving_mode \
  /leo/route_avoidance \
  /leo/signal_status \
  /semi/raceline_ctrl_cmd \
  /leo/raw_ctrl_cmd \
  /ctrl_cmd \
  /competition/ego_speed_kmh
```

LiDAR 토픽을 포함하면 ROS Bag 파일 크기가 빠르게 증가할 수 있으므로 저장 공간을 확인해야 합니다.

### 9.3 주행 상태 확인

```bash
rosnode list
```

```bash
rostopic hz /gps
```

```bash
rostopic hz /imu
```

```bash
rostopic hz /lidar3D
```

제어 명령 발행 여부:

```bash
rostopic hz /ctrl_cmd
```

통신이 정상이어도 실제 차량이 움직이지 않는다면 MORAI의 Ego Network 설정, UDP 포트, 차량 제어 모드, Signal Guard 상태를 함께 확인해야 합니다.

---

## 10. 주요 파일 안내

### 최종 주행 및 제어

| 파일 | 역할 |
|---|---|
| `unita_launch/launch/extreme.launch` | 최종 통합 자율주행 실행 |
| `unita_waypoint/scripts/raceline_test_follower.py` | Raceline 기반 주행 |
| `unita_waypoint/scripts/gps_waypoint_follower_local.py` | Leo Waypoint 추종 |
| `unita_waypoint/scripts/lidar_obstacles.py` | Raceline 주행용 LiDAR 장애물 인식 |
| `unita_waypoint/scripts/lidar_obstacles_local.py` | Leo 주행용 LiDAR 장애물 인식 |
| `leo_avoidance/scripts/route_avoidance_node.py` | 장애물 회피 경로 판단 |
| `leo_avoidance/scripts/ego_odometry_node.py` | 차량 Odometry 추정 |
| `leo_avoidance/scripts/driving_state_node_local.py` | 주행 위치 및 GPS 음영 상태 관리 |
| `leo_avoidance/scripts/blackout_udp_control_local.py` | MORAI UDP 차량 제어 |
| `leo_guard/scripts/signal_guard_node_local.py` | 신호등 판단 및 최종 제어 명령 통합 |
| `perception/scripts/traffic_light_test.py` | YOLO 신호등 인식 |

### 주요 설정 데이터

| 파일 | 역할 |
|---|---|
| `unita_waypoint/config/map_extreme_project.json` | 최종 대회 주행 경로 및 목표 속도 |
| `unita_waypoint/config/map_tunner_project.json` | Map Tuner 기본 편집 프로젝트 |
| `unita_waypoint/config/stoplines.json` | 정지선 데이터 |
| `unita_waypoint/config/raceline_layout.yaml` | Raceline 관련 설정 |
| `unita_waypoint/config/competition_speed_profile.yaml` | Raceline 주행 속도 설정 |
| `unita_waypoint/config/competition_speed_profile_local.yaml` | Leo 주행 속도 설정 |
| `leo_guard/config/intersections_local.json` | 교차로 및 신호 관련 데이터 |
| `leo_avoidance/config/route_lane_changes.json` | 차선 변경 가능 구간 데이터 |
| `sensor_bridge/config/sensors.yaml` | 센서 UDP 설정 |
| `control/config/competition_udp.yaml` | 차량 제어 UDP 설정 |
| `best1.pt` | YOLO 모델 가중치 |

### 시각화 및 경로 편집

| 파일 | 역할 |
|---|---|
| `unita_waypoint/launch/map_debug.launch` | 실시간 주행 지도 및 상태 시각화 |
| `unita_waypoint/launch/map_tunner.launch` | 주행 경로 및 속도 편집 |
| `unita_waypoint/MAP_TUNNER.md` | Map Tuner 사용 설명서 |

---

## 11. 실행 명령어 요약

| 목적 | 명령어 |
|---|---|
| 전체 자율주행 실행 | `roslaunch unita_launch extreme.launch` |
| ROS 주행 로그 활성화 | `roslaunch unita_launch extreme.launch log:=true` |
| RViz 포함 주행 | `roslaunch unita_launch extreme.launch start_rviz:=true` |
| Map Debug | `roslaunch unita_waypoint map_debug.launch` |
| Map Tuner | `roslaunch unita_waypoint map_tunner.launch` |
| ROS 토픽 목록 | `rostopic list` |
| ROS 노드 목록 | `rosnode list` |
| 전체 빌드 | `catkin_make` |

---

## 12. 주의사항

1. `extreme.launch`는 전체 주행 시스템을 실행하므로 동일한 센서 브릿지나 차량 제어 노드를 중복 실행하지 않도록 합니다.
2. MORAI와 ROS가 서로 다른 PC에 있다면 실제 네트워크 IP를 사용해야 합니다.
3. 센서 UDP 포트는 MORAI 설정과 `sensors.yaml`에서 일치해야 합니다.
4. 최종 주행 파일은 `map_extreme_project.json`이며 Map Tuner의 기본 파일과 구분해야 합니다.
5. `best1.pt`는 워크스페이스 최상위 경로에 위치해야 기본 `model_path` 설정과 일치합니다.
6. `build`, `devel`, 로그, Python 캐시 파일은 Git 저장 대상에서 제외합니다.
7. 본 프로젝트의 주행 설정은 MORAI 시뮬레이터 기반 대회 환경에 맞춰 개발되었으며, 실제 차량에서 사용하려면 별도의 안전 검증과 제어 파라미터 조정이 필요합니다.

---

## 13. 관련 문서

- [UNITA Launch 사용 설명서](src/unita_launch/README.md)
- [Map Tuner 사용 설명서](src/unita_waypoint/MAP_TUNNER.md)
- [Sensor Bridge 사용 설명서](src/sensor_bridge/README.md)
- [Control 사용 설명서](src/control/README.md)

---

## Team UNITASTAR

**인천대학교 자율주행 동아리 UNITA**

2026 창작모빌리티 경진대회 참가팀 **유니타스타(UNITASTAR)**

다양한 자율주행 대회 경험을 바탕으로 축적한 기술과 협업 경험을 하나의 통합 자율주행 시스템으로 구현하였습니다.
