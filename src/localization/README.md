# GPS + IMU localization

기존 `unita_waypoint/scripts/gps_waypoint_follower.py`의 위치 추정 로직을
독립 노드로 추출했습니다. 기존 follower/launch는 그대로 두었으며, 아직 이 노드의
출력을 구독하지 않습니다. 후속 planner/control 분리에서 pose를 연결할 수 있습니다.

```text
/gps (GPSMessage) ── UTM(zone 52) − map offsets ── x, y ─┐
/imu (Imu) ──────── normalized quaternion → yaw ─────────┤
                                                      └─ /localization/pose
```

| 구분 | 토픽 | 타입 / 의미 |
| --- | --- | --- |
| 입력 | `/gps` | `morai_msgs/GPSMessage` |
| 입력 | `/imu` | `sensor_msgs/Imu` |
| 출력 | `/localization/pose` | `geometry_msgs/PoseStamped`, map 좌표 x/y 및 yaw |
| 출력 | `/localization/valid` | `std_msgs/Bool`, 최신 센서 쌍의 사용 가능 여부 |
| 선택 출력 | `/tf` | `map → base_link`, `publish_tf:=true`일 때 |

## 실행

```bash
source ~/catkin_ws/devel/setup.bash
cd ~/unita_ws
catkin_make
source devel/setup.bash
roslaunch localization gps_imu_localization.launch
```

sensor_bridge는 별도로 실행합니다. 이 launch는 제어 명령을 발행하지 않습니다.

```bash
rostopic echo /localization/pose
rostopic echo /localization/valid
# map -> base_link TF를 이 노드가 담당할 경우에만 사용
roslaunch localization gps_imu_localization.launch publish_tf:=true
```

## 좌표와 유효성

- `x = UTM easting − GPSMessage.eastOffset`,
  `y = UTM northing − GPSMessage.northOffset`. 지도 offsets는 sensor_bridge에서
  설정해야 하며, 이 노드는 수신한 값을 그대로 사용합니다. 0인 경우 UTM 절대좌표가 됩니다.
- 2D 위치 추정입니다. z/roll/pitch는 0이며 고도·가속도를 적분하지 않습니다.
  EKF나 터널 내 관성 항법은 구현하지 않았습니다.
- GPS와 IMU가 차량 기준점에 정렬되어 있고, IMU yaw의 축이 map 축과 일치한다고
  가정합니다. GPS 안테나 위치 보정 및 UTM 자오선 수렴각 보정은 수행하지 않습니다.
  `yaw_offset`으로 고정 방향 차이(rad)를 조정할 수 있습니다.
- 기본 입력은 이 워크스페이스의 sensor_bridge입니다. `status > 0`을 GPS fix로
  해석합니다. 다른 GPS publisher는 status 의미를 확인하세요.
- 잘못된 좌표/쿼터니언, fix 없음, IMU orientation 미제공은 이전 상태를 무효화합니다.
  기본 1초 timeout 또는 GPS/IMU timestamp 차이 0.2초 초과 시에도 pose/TF 발행을
  중지하고 valid=false를 보냅니다. 센서가 복구되면 자동 재개합니다.
- timestamp가 0이면 수신 시각을 사용합니다. 그 외에는 GPS/IMU와 ROS가 동일한
  시간 기준을 사용해야 합니다. 출력은 두 센서 중 오래된 timestamp를 유지합니다.
- 소비 노드도 pose timestamp/valid 수신 timeout을 확인해야 합니다. 노드가 종료되면
  valid=false조차 새로 발행되지 않으며, 마지막 위치를 계속 사용하면 안 됩니다.
- TF 기본값은 false입니다. 다른 노드가 `odom → base_link`를 발행하는 구성에서는
  이 노드의 `map → base_link`를 동시에 켜지 마세요. 센서 장착 TF는 별도 설정입니다.

설정은 `config/gps_imu.yaml`에 있습니다. 기본 발행 주기는 20 Hz이고,
센서 주기·지연에 맞춰 `sensor_timeout`, `max_sensor_skew`를 조정할 수 있습니다.

## 파일 구성 및 테스트

- `src/unita_localization/estimator.py`: 좌표 변환, yaw 계산, 입력/시간 검증
- `scripts/gps_imu_localization.py`: ROS 구독, pose/valid 및 선택적 TF 발행
- `config/gps_imu.yaml`, `launch/gps_imu_localization.launch`: 설정 및 실행
- `test/test_estimator.py`: 투영 기준점, offsets, yaw, 센서 손실·복구 검증

```bash
catkin_make run_tests_localization
catkin_test_results build/test_results/localization
```
