# K-City Map Tuner

ROS 1 Noetic RViz에서 구간별 경로점과 속도를 편집하는 도구입니다. START→1이 S1,
1→2가 S2, …, 14→START가 S15입니다. 각 구간은 독립된 Raceline 1~3을
가지며, 초기 상태에는 `config/waypoints.csv`에서 가져온 Raceline 1만 있습니다.
새 차선을 추가하면 빈 Raceline이 생성됩니다.

## 사진과 waypoint 맞추기

기본 배경은 `config/kcity_map_sim.png`입니다. 별도 Python 창에서 사진을
waypoint에 맞춘 뒤 RViz에서 같은 위치를 사용할 수 있습니다.

```bash
source ~/unita_ws/devel/setup.bash
rosrun unita_waypoint map_waypoint_calibration.py
```

- 하늘색 선은 원본 waypoint이며 초록색 표시는 체크포인트입니다. 이 둘은 고정되어
  있고 사진만 움직입니다.
- 사진 안쪽을 드래그하면 이동합니다. 노란 모서리를 드래그하면 가로세로 비율을
  유지해 확대하거나 축소합니다. 파란 변 중앙을 드래그하면 해당 방향으로만
  늘어나거나 줄어듭니다. 분홍 원을 드래그하면 사진이 회전합니다.
- 마우스 휠로 화면을 확대하고 `전체 보기`로 돌아옵니다. `되돌리기` 또는
  `Ctrl+Z`로 직전 조작을 취소합니다.
- `RViz용 배치 저장` 또는 `Ctrl+S`를 누르면 `config/map_tunner_calibration.yaml`에
  사진의 네 모서리 좌표를 저장합니다. 실행 중인 Map Tuner도 약 1초 간격으로
  변경을 감지하여 지도를 갱신합니다. waypoint 좌표는 변경되지 않습니다.

RViz의 `K-City 컬러 배경사진` 표시는 같은 배치 좌표로 원본 색상을 표시합니다.
기본 `Map` 표시는 컬러 표시에 문제가 있을 때를 위한 회색 대체 화면입니다.

## 실행

```bash
source ~/unita_ws/devel/setup.bash
roslaunch unita_waypoint map_tunner.launch
```

ROS master가 없다면 별도 터미널에서 `roscore`를 먼저 실행하세요. MORAI나 GPS
센서 없이도 편집할 수 있습니다. 이전 RViz 창이 열려 있다면 닫고 위 명령으로
새 창을 실행해야 고정 편집 패널이 나타납니다.

## 화면 읽기

- 왼쪽 **K-City 경로 편집** 패널이 조작부입니다. RViz의 `Displays` 속성 목록은
  편집 버튼이 아닙니다.
- **파란 점**은 체크포인트, **하늘색 선**은 다른 구간의 기존 경로입니다.
  선택한 구간의 Raceline 1은 **빨간 선**, Raceline 2는 **파란 선**,
  Raceline 3은 **초록 선**으로 보입니다. **노란 점**은 현재 선택한 경로점입니다.
- 사진과 경로가 세로로 보이거나 화면 밖으로 나가면 맨 위의
  **↶ 90° 회전 · 전체 지도 보기**를 누르세요. 사진과 waypoint가 함께 가로로
  표시되며 전체 지도가 화면 안에 들어옵니다. RViz에서 마우스 휠로 확대할 수
  있습니다.

## 경로 편집

1. **구간 선택** 목록에서 S1~S15를 고릅니다. 지도에서 파란 체크포인트를
   `Interact`로 눌러도 해당 체크포인트에서 시작하는 구간을 고를 수 있습니다.
2. **＋ 차선 추가**는 현재 구간에 빈 Raceline 2, 이후 빈 Raceline 3을 만듭니다.
   **－ 마지막 차선 삭제**는 번호가 가장 큰 차선과 그 점들을 삭제합니다.
   Raceline 1은 삭제할 수 없습니다. `실행 취소`로 마지막 편집을 되돌릴 수 있습니다.
3. **편집할 Raceline** 목록에서 경로를 고릅니다. 구간과 Raceline은 각각
   독립적이므로 현재 선택한 경로의 점만 바뀝니다.
4. **＋ 점 추가**를 누르고 지도에서 좌클릭하면 점이 추가됩니다. 기존 경로에는
   가장 가까운 선분 뒤로 삽입하며, 빈 경로에는 클릭한 순서대로 쌓입니다.
5. **점 선택**을 누르고 기존 점 가까이 좌클릭하면 노란 선택 점이 나타납니다.
   도구가 자동으로 드래그 모드로 바뀌므로 노란 점을 마우스로 끌어 상하좌우로
   이동할 수 있습니다. 다른 점을 선택하려면 **점 선택**을 다시 누르세요.
   선택한 점의 속도는 텍스트 입력 후 **속도 적용 (km/h)**로 바꿉니다.
6. **시작점·끝점 선택**을 누르고 현재 Sector/Raceline의 두 점을 차례로
   좌클릭하세요. 두 점 사이가 노란색으로 표시됩니다. 속도 값을 입력하고
   **구간 전체에 적용**을 누르면 양 끝을 포함한 모든 점이 같은 속도가 됩니다.
   선택 순서는 앞뒤 어느 쪽이든 됩니다. 허용 속도는 0~200 km/h입니다.
7. **사이 점 개수**에 시작점·끝점을 제외한 점 개수를 입력하고 **연속 점찍기**를
   누르세요. 지도에서 시작점과 끝점을 차례로 좌클릭하면 두 위치 사이에 균등하게
   점을 생성하여 현재 Raceline 끝에 추가합니다. 예를 들어 `10`을 입력하면 양 끝을
   포함해 총 12점이 추가됩니다. 빈 Raceline과 기존 Raceline 모두 사용할 수 있으며,
   기존 점은 유지됩니다. 기존 경로의 끝부터 이어 그리려면 끝점 위치를 시작점으로
   클릭하세요. 새 점의 속도는 시작 위치에 가장 가까운 기존 점에서 가져오며 높이는
   양 끝 근처 점의 높이를 보간합니다. **실행 취소**로 한 번에 되돌릴 수 있습니다.
8. **JSON 저장**을 누르면 `config/map_tunner_project.json`에 기록됩니다.
   저장하지 않은 편집이 있으면 버튼에 ●가 표시됩니다. 원본
   `config/waypoints.csv`는 변경하지 않습니다.

추가 점의 `z`는 원본 경로에서 가장 가까운 waypoint의 값을 복사합니다.
지도에서 점을 움직이면 `x`, `y`만 바뀝니다. JSON의 각 점에는
`id`, `x`, `y`, `z`, `speed_kmh`가 저장됩니다. 출력 경로는
`project_file:=/absolute/path/project.json`으로 바꿀 수 있습니다.

## 사진 위치 보정

위의 별도 Python 보정 창에서 사진을 이동·회전·늘린 뒤 저장하세요. 사진 버전이
달라 도로 구조 자체가 다른 구간은 사진 배치만으로 경로가 일치하지 않을 수
있습니다. 그때는 해당 구간의 waypoint를 따로 편집해야 합니다.

편집 노드는 `/map_tunner/map`, `/map_tunner/markers`,
`/map_tunner/interactive/update`, `/map_tunner/status`를 발행합니다.
`competition_udp_tracking.launch`는 저장된 JSON의 각 Sector Raceline 1을
순서대로 이어 주행합니다. 편집 후 **JSON 저장**을 누르고 주행 launch를 다시
시작해야 새 경로가 적용됩니다. 원본 `waypoints.csv`는 그대로 둡니다.
일반 `competition_udp_tracking.launch`는 편집기 속도 값을 적용하지 않고 기존
곡률 기반 속도 계획과 `competition_speed_profile.yaml`의 상한을 사용합니다.
`test_map_raceline.launch`는 Raceline 1의 점별 `speed_kmh`를 추가 상한으로
사용합니다. 차선 전환 중에도 동일한 점별 상한을 적용합니다.

## 읽기 전용 주행 지도

주행 launch와 같은 ROS master에서 별도 터미널로 실행합니다.

```bash
roslaunch unita_waypoint map_debug.launch
```

편집기와 같은 사진, 저장된 경로, Sector Checkpoint를 표시합니다. 왼쪽에는
실제 속도, 지도 좌표, 헤딩, 현재 Sector를 표시하고 지도에는 차량 위치와
진행 방향을 노란 화살표로 그립니다. 유효한 follower 상태에 담긴 제어 위치와
헤딩을 우선 사용하여 우선 차선 판정과 차량 표시를 같은 시점에 맞춥니다.
follower 위치가 없으면 `/localization/pose`와 `/localization/valid`를 사용하고,
속도는 `/competition/ego_speed_kmh`를 사용합니다. 주행 launch를
`start_rviz:=false`로 시작하면 RViz 창이 하나만 열립니다. 위치가 유효하지
않거나 1초 넘게 갱신되지 않으면 차량 표시를 지웁니다.

Map Debug는 `/home/unita/unita_ws/src/unita_waypoint/config/map_tunner_project.json`을
고정으로 읽어 모든 Raceline의 선과 점을
표시합니다. 자동 차선 재생성을 하지 않으므로 수동 연장한 점도 유지됩니다.
Map Tuner에서 JSON을 저장하면 실행 중인 Map Debug도 약 1초 안에 경로를
갱신합니다. 실행 로그에는 읽은 파일의 절대 경로와 Sector/Raceline별 점 개수가
표시됩니다. 주행 노드가 발행한 `lane_geometry`는 장애물 필터에만 사용하며
저장된 지도 표시를 덮어쓰지 않습니다. `project_file` 인자로 다른 파일을
지정해도 Debug의 지도 파일은 위 경로로 고정됩니다.

## 기준점으로 3차선 경로 생성

Sector 11의 Raceline 1은 `config/waypoints.csv`의 해당 구간 XYZ 좌표를 그대로
사용합니다. 점별 `speed_kmh`는 프로젝트 JSON 값을 유지합니다. 기준점
`(63.47, 146.13)`에 가장 가까운 Raceline 1 점이 차선 1의 기준입니다.
남쪽으로 주행하는 이 구간에서 왼쪽은 지도 +X 방향입니다.

기준점에 가장 가까운 점의 X 좌표를 **처음 초과하는 다음 점**부터 Raceline 2와
Raceline 3을 만듭니다. 두 경로는 각각 Raceline 1보다 +X 방향으로
`raceline_width`, `2 × raceline_width`만큼 이동한 차선 중앙입니다.
점 개수와 대응 점의 Y/Z 좌표, 연속 점 사이 간격은 Raceline 1과 같습니다.
두 경로는 `y >= -122.90`인 마지막 점에서 함께 끝나고, 그 뒤에는
Raceline 1만 이어집니다. Raceline 1의 앞뒤 점과 다른 Sector는 수정하지 않습니다.

`map_tunner.launch`는 저장된 프로젝트의 차선 폭을
기본으로 표시합니다. 저장된 값이 없으면 공통 설정
`config/raceline_layout.yaml`의 `raceline_width: 3.5` m를 사용합니다.
현재 저장된 프로젝트의 폭은 3.2 m입니다. 폭을 바꿔 확인하려면:

```bash
roslaunch unita_waypoint map_tunner.launch raceline_width:=3.2
```

폭을 변경하면 CSV의 Raceline 1을 기준으로 Raceline 2/3을 다시 만듭니다.
Map Tuner에서 **JSON 저장**을 누르면 일반 주행에서도 새 경로를 사용합니다.
저장 전 수동으로 바꾼 Raceline 2/3 위치는 폭을 변경해 다시 생성할 때
초기화됩니다. 원본 `waypoints.csv`는 수정하지 않습니다.
Map Debug의 경로 폭을 바꾸려면 Map Tuner에서 수정하고 JSON을 저장하세요.

Map Debug 왼쪽의 **현재 Sector**와 **추종 Raceline**은 주행 follower의
`/waypoint_debug/tracking_state`를 받아 표시합니다. Raceline 번호는 차량이
물리적으로 위치한 차선의 추정값이 아니라 현재 추종하는 목표 경로입니다.
일반 주행은 Raceline 1이 기본이며, `test_map_raceline.launch`는 저장된 경로가
있는 Sector 11 구간에서 checkpoint 11까지 Raceline 3을 우선하고,
checkpoint 11 이후에는 Raceline 1로 복귀합니다. 장애물과 후방 간격에 따라 추월·복귀를
판단하며, 패널에는 현재 추정 차선 → 목표 차선과 우선 차선이 함께 표시됩니다.
끝선은 JSON의 S11 끝점 Y=-217.855053961277이며 지도에 흰색 가로선으로 표시합니다.
차량 뒤쪽 모서리까지 끝선의 남쪽으로 지나야 R1 우선으로 바뀝니다.
패널의 **CP11 끝선**에서 뒤쪽 Y, 끝선 Y, 후미 통과 여부를 확인할 수 있습니다.
수동 연장한 R3도 주행 경로에 포함됩니다. 주행은 차선 폭 파라미터와 무관하게
저장된 JSON 좌표를 그대로 사용합니다. 경로 재생성은 Map Tuner에서 수행하고
JSON을 저장합니다. 주행 launch의 차선 폭은 장애물 인식 범위에만 적용됩니다.
추종 상태가 1초 이상 수신되지 않거나 정지 상태이면 Raceline은 대기 중으로 표시합니다.
