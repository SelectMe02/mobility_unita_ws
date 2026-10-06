# Raceline 동적 차량 시나리오

`UNITA_raceline_dynamic_traffic.json`은 `2026_molit_comp_sample_scene.json`의
NPC 생성기 4개를 추출한 `R_KR_PR_K-city_2025` 테스트 시나리오입니다.

- 생성 위치, 차량 종류, 출발/도착 링크, 생성 간격, 목표 속도(20/40/50/60 km/h)는 원본과 같습니다.
- 생성 카운트와 대기 시간을 0으로 초기화하고 `isSpawning`을 활성화했습니다.
- 저장된 NPC 9대, 보행자, 정적 물체, 신호 강제 설정, GPS 음영 영역은 제외했습니다.
- Ego의 원본 출발 위치/방향은 유지하고 초기 속도를 0, 기어를 D(4)로 설정했습니다.
- 네트워크 파일은 기존 `UNITA_tunnel_gps_blackout_MN.json` 형식을 사용해
  Cmd Control과 Ego Vehicle Status만 UDP로 활성화했습니다.

## 실행

1. MORAI의 **Edit → Scenario → Load Scenario**에서
   `UNITA_raceline_dynamic_traffic`을 선택하고 **네트워크 연결 데이터 로드**도 선택합니다.
   JSON과 `EgoNetwork/UNITA_raceline_dynamic_traffic_MN.json`은 같은 시나리오 디렉터리에 있어야 합니다.
2. Ego Network에서 Cmd Control의 UDP 수신이 `127.0.0.1:9093`,
   Ego Vehicle Status의 UDP 송신 목적지가 `127.0.0.1:9096`인지 확인합니다.
   시나리오 로드 후 연결이 해제되면 다시 연결하고 시뮬레이션 일시정지를 해제합니다.
3. 센서 설정은 별도로 로드합니다. 기존
   `SaveFile/Sensor/25.S4.MolitComp03/UNITA_LOCAL_SENSORS_FIXED.json`의 포트와
   LiDAR 장착 위치가 현재 launch와 일치합니다.
   GPS 목적지 9230, IMU 9220, LiDAR 9210이며 IP는 모두 `127.0.0.1`입니다.
4. 다른 주행 launch를 종료한 뒤 실행합니다.

```bash
source /home/unita/unita_ws/devel/setup.bash
roslaunch unita_launch test_map_raceline.launch morai_ip:=127.0.0.1
```

시나리오 JSON은 센서 설정과 ROS launch를 자동 실행하지 않습니다.
현재 launch는 `/imu`를 사용하고, 유효한 GPS·방향·차량 상태·LiDAR 장애물 입력을 기다립니다.
입력이 누락되어 정지하면 다음 명령을 **각각** 실행해 상태를 확인합니다.

```bash
rostopic echo -n 1 /waypoint_debug/tracking_state
rostopic echo -n 1 /control/udp_status
```

이 파일은 동적 교통 생성용이며 항상 추월이 발생하는 시험은 아닙니다.
현재 회피 경로의 속도 제한보다 빠른 NPC는 추월하지 못할 수 있습니다.
JSON 구조와 포트 일치는 파일 수준에서 검증했으며, 실제 NPC 생성 및 주행은
MORAI에서 확인해야 합니다.
