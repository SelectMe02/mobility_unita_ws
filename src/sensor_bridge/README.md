# MORAI UDP → ROS sensor bridge

`~/catkin_ws`의 기존 `morai_msgs`를 사용합니다. 이 패키지는 센서 수신만
수행하며 차량 제어 명령을 보내지 않습니다.

3-camera 구성에서는 `sensor_bridge.launch`에 `enable_rear_camera:=false`를
지정하면 네 번째 카메라 수신기만 제외할 수 있습니다. 단독 실행 기본값은
기존처럼 true입니다. 현장 통신 진단은 `unita_launch/field_check.launch`를
사용하며 [현장점검 절차](../unita_launch/README.md#field-check--5분-현장점검-절차)를
참고하세요.

| 센서 | MORAI Host | ROS PC Destination | ROS 토픽 | 메시지 |
| --- | ---: | ---: | --- | --- |
| Camera-1 | 9101 | 9201 | `/camera/image/front` | `sensor_msgs/Image` (`bgr8`) |
| Camera-2 | 9102 | 9202 | `/camera/image/left` | `sensor_msgs/Image` (`bgr8`) |
| Camera-3 | 9103 | 9203 | `/camera/image/right` | `sensor_msgs/Image` (`bgr8`) |
| 추가 카메라 (rear) | 9104 | 9204 | `/camera/image/rear` | `sensor_msgs/Image` (`bgr8`) |
| Lidar3D-4 | 9110 | 9210 | `/lidar3D` | `sensor_msgs/PointCloud2` |
| IMU-5 | 9120 | 9220 | `/imu` | `sensor_msgs/Imu` |
| GPS-6 | 9130 | 9230 | `/gps` | `morai_msgs/GPSMessage` |

MORAI Sensor Network에서 UDP를 선택하고 **Destination IP를 ROS PC의 IP**로
설정합니다. 브리지는 Destination 포트에 bind하며, Host 포트를 bind하거나
센서에 요청 패킷을 보내지 않습니다. 기본 `bind_ip=0.0.0.0`은 로컬의 모든
인터페이스에서 수신한다는 의미이며, MORAI의 Destination IP에 넣는 값이 아닙니다.
`source_ip`를 지정하면 해당 송신 IP의 데이터만 수신합니다.

## 빌드 및 실행

```bash
source ~/catkin_ws/devel/setup.bash
cd ~/unita_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

LiDAR를 포함하려면 Velodyne 패키지가 필요합니다.

```bash
sudo apt-get install ros-noetic-velodyne
roslaunch sensor_bridge sensor_bridge.launch
```

LiDAR 드라이버 설치 전에는 카메라 4개, GPS, IMU만 실행할 수 있습니다.

```bash
roslaunch sensor_bridge sensor_bridge.launch enable_lidar:=false
```

기본 LiDAR 설정은 **VLP16, 600 RPM(10 Hz)**이며 실제 SIM 모델을 확인한 값이
아닙니다. SIM의 모델 및 Rotation Rate에 맞춰 실행하세요.

```bash
# HDL32, Rotation Rate 10 Hz 예시
roslaunch sensor_bridge sensor_bridge.launch lidar_launch:=32e_points.launch lidar_rpm:=600

# HDL64 S3 예시 (MORAI 문서의 64e_s3-xiesc.yaml 보정 파일 확인 필요)
roslaunch sensor_bridge sensor_bridge.launch lidar_launch:=64e_S3.launch lidar_rpm:=600
```

`lidar_rpm = Rotation Rate(Hz) × 60`입니다. 지원 launch는 Velodyne 1.7.0의
`VLP16_points.launch`, `32e_points.launch`, `64e_S3.launch`와 대조했습니다.
LiDAR 포트·토픽은 `lidar_port`, `lidar_topic` 인자로 변경합니다.
나머지 센서의 포트·토픽·frame_id는 `config/sensors.yaml`에서 변경합니다.
YAML의 LiDAR 항목은 매핑 참고용이고 Velodyne 노드 설정은 launch 인자가 적용됩니다.

## 카메라 4개 RViz 확인

추가 카메라는 후방으로 가정해 `camera_rear`로 등록했습니다. Host Sensor Port는
`9104`, Destination Port는 **9204**로 맞추고 Destination IP를 ROS PC IP로
설정한 뒤 센서 UDP 송신을 연결합니다. 같은 PC이면 `127.0.0.1`을 사용합니다.
다른 방향이면 `config/sensors.yaml`의 이름·토픽·frame_id와 launch 및 RViz
설정의 대응 항목을 함께 변경합니다.

기존 sensor_bridge를 종료하고 다시 실행하면 네 번째 수신기도 함께 켜집니다.
카메라 전용 RViz는 센서 브리지가 실행 중인 상태에서 별도 터미널로 엽니다.

```bash
source ~/unita_ws/devel/setup.bash
roslaunch unita_launch camera_visualization.launch
```

Front/Left/Right/Rear camera의 `rviz/Image` 패널을 표시하며 Transport Hint는
`raw`입니다. 주행용 `waypoint_tracking.rviz`에도 네 개의 Image display를 추가했습니다.
패널이 가려져 있으면 Displays에서 해당 카메라를 체크하고 Panels 메뉴에서
이미지 패널을 켭니다. 제목 표시줄을 드래그해 원하는 위치에 배치할 수 있습니다.
Image display는 CameraInfo와 센서 장착 TF 없이 2D 이미지를 확인할 수 있습니다.
여기서 raw는 JPEG UDP를 디코딩한 `bgr8` ROS 이미지이며 JPEG 압축 이전의
센서 데이터가 복원된다는 의미는 아닙니다.

```bash
rostopic info /camera/image/rear
rostopic hz /camera/image/rear
rostopic echo -n 1 --noarr /camera/image/rear
```

토픽 타입은 `sensor_msgs/Image`, frame_id는 `camera_rear_optical`입니다.
아무 영상도 없으면 SIM Destination Port 9204, Connect 및 토픽 발행 주기를 확인합니다.

## GPS 지도 좌표와 시간

GPS UDP는 NMEA RMC/GGA이고 **eastOffset/northOffset을 전송하지 않습니다**.
위도·경도·고도는 유효한 GGA에서 발행하며 `status`에는 GGA fix quality를 넣습니다.
RMC는 GGA와 중복되는 좌표여서 별도 발행하지 않습니다. 체크섬이 있으면 검증하고,
fix가 없거나 형식이 잘못된 문장은 버립니다.

기존 `gps_waypoint_follower.py`는 UTM 좌표에서 offsets를 빼므로, 주행 전에
현재 MORAI 지도의 실제 보정값을 전달해야 합니다. 기본값은 0이며 경고가 출력됩니다.

```bash
# EAST_OFFSET / NORTH_OFFSET에는 사용 중인 지도의 실제 숫자를 지정
roslaunch sensor_bridge sensor_bridge.launch \
  east_offset:="$EAST_OFFSET" north_offset:="$NORTH_OFFSET"
```

기본 메시지 시각은 ROS 수신 시각입니다. `use_sensor_stamp:=true`는 카메라 및
timestamp가 포함된 IMU의 송신 시각을 사용합니다. `/clock`은 이 패키지에서 발행하지
않으므로 시뮬레이션 시간과 ROS 시간을 맞춘 경우에만 이 옵션을 사용하세요.
frame_id를 부여하지만 센서 장착 위치의 TF나 카메라 보정 정보는 생성하지 않습니다.

## 지원 패킷과 검증

- 카메라: timestamp가 있는 `MOR` JPEG 패킷, 가변 길이 또는 65000-byte padding.
  분할 순서·timestamp·길이를 검사하고 불완전한 프레임은 버립니다. `BOX`는 무시합니다.
  timestamp 없는 구버전 및 Depth 전용 데이터는 지원 대상이 아닙니다.
- IMU: `#IMUData$`, 107-byte(timestamp 없음) 및 115-byte(timestamp 포함).
- LiDAR: 공식 Velodyne 드라이버의 패킷 수신·보정·PointCloud2 변환 사용.

```bash
catkin_make run_tests_sensor_bridge
catkin_test_results build/test_results
rostopic hz /camera/image/front
rostopic hz /lidar3D
rostopic echo -n 1 /gps
rostopic echo -n 1 /imu
```

파서 테스트 7개와 별도 UDP 포트/ROS master를 사용한 카메라 3개·GPS·IMU의
통합 테스트를 수행했습니다. LiDAR는 드라이버 미설치로 실수신 검증을 하지 않았습니다.
추가 카메라를 포함한 4개 수신기도 별도 ROS master와 시험 JPEG UDP로 검증했으며,
각 토픽의 이미지 크기·bgr8 인코딩·frame_id와 영상 분리를 확인했습니다.

## 참고 자료

- [MORAI-NetworkModule](https://github.com/MORAI-Autonomous/MORAI-NetworkModule/tree/24.R2.0)
  (`78e88558588451bdf9a10baf04d575c9aa3e8587`의 Camera/GPS/IMU 정의 참고)
- [MORAI 센서 통신 프로토콜](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35)
- [MORAI ROS 메시지 beta_drive](https://github.com/MORAI-Autonomous/MORAI-ROS_morai_msgs/tree/beta_drive)
- [Velodyne ROS 1.7.0](https://github.com/ros-drivers/velodyne/tree/1.7.0)
- [RViz Noetic Image display](https://github.com/ros-visualization/rviz/blob/noetic-devel/src/rviz/default_plugin/image_display.cpp)
