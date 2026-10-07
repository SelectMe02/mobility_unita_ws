# 신호등 정지선·노란불 판단

## 맵 데이터

`stoplines.json`은 다음 명령으로 생성했습니다. 입력은 MORAI MGeo의 지도 로컬
좌표(`local_origin_in_global = [302595, 4124145, 0]`)입니다.

```bash
python3 src/unita_waypoint/scripts/extract_stoplines.py \
  ~/Downloads/node_set.json ~/Downloads/link_set.json
```

출력은 `src/unita_waypoint/config/stoplines.json`입니다. 각 항목은 `idx`, `x`,
`y`, `traffic_light_id`, `from_link_idxs`, `to_link_idxs`를 포함합니다. 원본에서
`on_stop_line == true`이고 신호등 ID가 있는 노드만 포함합니다. 같은 신호등의
차로별 노드는 따로 유지합니다.

현재 파일에서는 정지선 노드 69개, `from_node_idx` 링크 참조 124개,
`to_node_idx` 링크 참조 70개입니다. 69개 노드 모두 양쪽 링크가 있습니다.
이 숫자만으로 정지선이 링크의 시작 또는 끝이라고 판단하지 않습니다.

## 주행 연결

`leo_guard/launch/signal_tracking.launch`는 기존 경로 추종 PID의
`/leo/raw_ctrl_cmd`와 `/leo/speed_limit_kmh` 경로를 사용합니다. 신호 입력은
Leo의 전방/상향 카메라 전환 인식 `/perception/traffic_lights`입니다. 이 토픽에는
신호등 ID가 없으므로 `config/intersections.json`의 경로 연관 ID, 지정된
`signal_camera`, 검토된 `signal_roi`와 `signal_min_width`를 함께 사용합니다.
현재 조사 파일에서 신호 제어 교차로 5개가 검토 상태입니다. 카메라·검출이
맞지 않거나 모호하면 정지합니다. Leo의 우회전 일시정지·LiDAR 양보 판단과
미제어 교차로 속도 제한은 기존 로직을 유지합니다.

직진 경로의 노란불에서는 앞범퍼에서 정지선까지 진행방향 투영 거리 `d_m`와
속도 `v_mps`를 계산합니다. 경로 조사 파일이 지정한 차로별 노드를 사용하고,
횡방향 오차가 크면 진행을 허용하지 않습니다. 마지막으로 관측한 초록불의
촬영 시각을 노란불 시작의 보수적
하한으로 사용합니다. 이전 초록불 관측이 없거나 오래됐으면 통과 판단을
하지 않습니다. `can_go`이면 통과, 아니면 목표 감속도 기반 속도 상한과
Leo의 기존 정지 여유를 함께 PID에 전달합니다. 딜레마 구간, 빨간불,
신호 불명은 정지 쪽으로 결정합니다.

설정값은 `src/leo_guard/src/leo_guard/yellow_stop.py`의 상단 블록에 있습니다.
`FRONT_OFFSET_M`, `A_BRAKE_MPS2`, `A_MAX_MPS2`, `T_DELAY_S`,
`YELLOW_DURATION_S`, `GO_MARGIN_S`는 모두 임시값이므로 실측해야 합니다.
매 제어 프레임 `/leo/signal_status`와 ROS 로그에 거리, 속도, 신호,
`can_go`, `can_stop`, 최종 판단이 출력됩니다.

## 시뮬레이터에서 확인 필요

1. 정지선에 앞범퍼를 맞춰 세운 뒤 `/leo/signal_status`의 `d_m`이 0에 가까운지
   확인합니다. 차이가 있으면 GPS 기준점과 `FRONT_OFFSET_M`을 측정합니다.
2. 직진 차로에서 `lateral_m`이 허용폭 안에 있고 옆 차로 노드가 선택되지 않는지
   확인합니다. 곡선 접근로는 투영 오차를 별도로 확인합니다.
3. `/localization/pose`와 `/competition/ego_status`의 좌표를 같은 시각에
   비교합니다. 원시 Ego UDP `position`에는 코드상 좌표 변환이 없습니다.
4. 노란불 첫 인식부터 빨간불 전환까지의 실제 시간, 인식 지연, 최대 제동 및
   PID 감속 궤적을 측정한 뒤 상단 상수를 수정합니다.
5. 기존 ROI의 현장 영상 검토 기록을 기준으로 정지선 앞 정지, 통과,
   딜레마, 신호 미검출, 빨간불 전환을 각각 재현합니다.

`leo` 브랜치 커밋에는 GPS 음영구역 주행 코드가 없습니다. 현재
`local-state-2026-10-05` 작업 트리에는 선택형
`leo_avoidance/launch/signal_tracking_avoidance.launch`가 추가되었습니다.
LiDAR ICP + Ego 오도메트리로 위치를 전환하면서 기존 Pure Pursuit/PID를
재사용합니다. 일반 `signal_tracking.launch`는 GPS가 끊기면 제동합니다.
