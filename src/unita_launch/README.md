# unita_launch — 주행 실행 및 RViz 시각화

기존 waypoint follower와 GPS+IMU localization을 조합해 기준 경로, 실제 궤적,
차량 방향과 경로 오차를 표시합니다.

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
