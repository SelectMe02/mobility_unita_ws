"""Shared live avoidance settings and acknowledgements for the RViz panel."""
import json
import math
import os
from pathlib import Path
import tempfile
import time

import rospy
from std_msgs.msg import String


# name: (owning node, default, planner/detector attribute, unit, description)
SETTINGS = {
    'cluster_min_size_m': ('lidar_obstacles', .2, 'cluster_min_size_m', 'm', '군집의 긴 XY 폭 최소값 · 미만이면 제외'),
    'cluster_max_size_m': ('lidar_obstacles', 24., 'cluster_max_size_m', 'm', '군집의 긴 XY 폭 최대값 · 초과하면 군집 전체 제외'),
    'front_range_m': ('lidar_obstacles', 55., 'front_range', 'm', '후륜 축 기준 전방 점군 범위'),
    'rear_range_m': ('lidar_obstacles', 55., 'rear_range', 'm', '후륜 축 기준 후방 점군 범위'),
    'side_range_m': ('lidar_obstacles', 12., 'side_range', 'm', '차량 좌우 각각의 점군 범위'),
    'dynamic_speed_threshold_mps': ('lidar_obstacles', .8, 'dynamic_speed_threshold_mps', 'm/s', '3회 이상 관측 시 이 속도 초과는 동적, 이하는 정적'),
    'obstacle_min_height_m': ('lidar_obstacles', .22, 'obstacle_min_height_m', 'm', '노면 미확인·IMU 누락 시 기준 노면 대비 높이 하한 (초과)'),
    'obstacle_max_height_m': ('lidar_obstacles', 2.6, 'obstacle_max_height_m', 'm', '장애물 높이 상한 (미만) · 노면 확인 시 해당 평면 기준'),
    'ground_plane_tolerance_m': ('lidar_obstacles', .04, 'ground_plane_tolerance_m', 'm', '확인된 노면 평면의 점 제거 허용오차 (초기값, 기록 후 조정)'),
    'imu_max_skew_s': ('lidar_obstacles', .05, 'imu_max_skew_s', 's', '점군 시각과 자세 보정용 IMU의 최대 시간 차이'),
    'ground_max_slope_deg': ('lidar_obstacles', 20., 'ground_max_slope_deg', '°', '노면으로 인정하는 최대 기울기'),
    'ground_reference_z_m': ('lidar_obstacles', 0., 'ground_reference_z_m', 'm', '보정 점군의 기준 노면 높이 · 음수 허용 · 장착 설정 확인 필요'),
    'track_confirm_time_s': ('lidar_obstacles', .2, 'track_confirm_time_s', 's', '작은 장애물 추적 확인에 필요한 관측 지속 시간 (최소 3회)'),
    'obstacle_horizon_m': ('gps_waypoint_follower', 45., 'obstacle_horizon_m', 'm', '현재 경로를 따라 장애물 중심을 판단하는 거리'),
    'avoidance_margin_m': ('gps_waypoint_follower', .35, 'margin', 'm', '차량과 장애물 사이 안전 여유'),
    'lane_change_gap_m': ('gps_waypoint_follower', 3., 'lane_change_gap_m', 'm', '차선 변경 시 지나는 차선의 앞·뒤 차량과 범퍼 사이 최소 간격 · 변경 중 침범하면 취소'),
    'lane_change_lateral_accel_mps2': ('gps_waypoint_follower', 1.5, 'lane_change_lateral_accel', 'm/s²', '차선 변경 중 최대 횡가속도 · 작을수록 길고 완만함 (고속에서 길이 결정)'),
    'lane_change_max_heading_deg': ('gps_waypoint_follower', 10., 'lane_change_max_heading_deg', '°', '차선 변경 중 차선 대비 최대 각도 (저속에서 길이 결정)'),
    'avoidance_reaction_s': ('gps_waypoint_follower', .4, 'reaction_s', 's', '감속·충돌 예측에 사용하는 반응 시간'),
    'avoidance_decel_mps2': ('gps_waypoint_follower', 3., 'decel', 'm/s²', '장애물 감속 계산에 사용하는 감속도'),
    'avoidance_brake_limit': ('gps_waypoint_follower', .35, 'avoidance_brake_limit', '0..1', '먼 예상 충돌 감속 시 브레이크 상한 · 긴급 제동은 제한 없음'),
    'avoidance_clear_hold_s': ('gps_waypoint_follower', .4, 'clear_hold_s', 's', '정지 해제 전 안전 판단 지속 시간 · 새 점군 3회도 필요'),
    'restart_accel_rate_per_s': ('gps_waypoint_follower', .5, 'restart_accel_rate', '1/s', '액셀 증가율 제한 · 브레이크는 즉시 적용'),
    'sector11_min_speed_kmh': ('gps_waypoint_follower', 45., 'sector_min_speed_kmh', 'km/h', 'S11 최저 속도 · 바로 앞 장애물이 없으면 정지하지 않음 (파일 속도 이하)'),
    'lane_change_speed_boost_kmh': ('gps_waypoint_follower', 10., 'lane_change_speed_boost_kmh', 'km/h', 'S11 차선 변경 중 목표 속도 = 현재 속도 + 이 값 (파일 속도 이하)'),
    'lane_change_cancel_scans': ('gps_waypoint_follower', 3., 'cancel_confirm_scans', '회', '차선 변경 취소에 필요한 연속 감지 횟수 (다가오는 3 m 내 차량)'),
    'lane_change_cancel_max_progress': ('gps_waypoint_follower', .6, 'cancel_max_progress', '0..1', '이 진행률을 넘은 차선 변경은 취소하지 않고 완료'),
    'route_max_error_m': ('leo_signal_guard', 3.75, 'route_max_error_m', 'm', '신호 guard 경로 이탈 허용 거리 (터널 밖) · 넘으면 정지 후 자동 재arm 대기'),
    'blackout_route_max_error_m': ('leo_signal_guard', 7., 'blackout_route_max_error_m', 'm', '신호 guard 경로 이탈 허용 거리 (터널 GPS 음영 구간)'),
    'pass_speed_kmh': ('leo_waypoint_follower', 20., 'pass_speed_kmh', 'km/h', 'Leo 장애물 추월·차선 변경 중 속도 상한'),
    'arc_lane_change_max_kmh': ('leo_waypoint_follower', 15., 'arc_lane_change_max_kmh', 'km/h', '이 속도 이하에서는 최소 회전반경 원호로 차선 변경'),
    'route_pass_speed_kmh': ('leo_route_avoidance', 20., 'pass_speed_kmh', 'km/h', 'GPS 경로 정적 장애물 반대 차선 추월 중 속도 상한'),
    'ftg_no_return_speed_mps': ('leo_driving_state', 2., 'no_return_speed_mps', 'm/s', '터널 전방 LiDAR 반사 없음(출구 등) 서행 속도'),
    'ftg_wall_min_clearance_speed_mps': ('leo_driving_state', 2., 'wall_min_clearance_speed_mps', 'm/s', '터널 우측 벽 최소 간격 미만일 때 속도 상한'),
    'ftg_near_obstacle_speed_mps': ('leo_driving_state', 2., 'near_obstacle_speed_mps', 'm/s', '터널 장애물이 정지거리 안일 때 속도 상한'),
    'ftg_wall_follow_min_speed_mps': ('leo_driving_state', 5., 'wall_follow_min_speed_mps', 'm/s', '터널 벽 추종 중 최소 속도 상한 (경로 방향 불신 시)'),
}

PERSISTENCE_DIR = Path(__file__).resolve().parents[1] / 'config'


def validated_value(name, value):
    if name not in SETTINGS:
        raise ValueError('수정할 수 없는 파라미터입니다: ' + str(name))
    if isinstance(value, bool):
        raise ValueError('숫자를 입력하세요.')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('숫자를 입력하세요.')
    if name == 'ground_reference_z_m':
        if not math.isfinite(number) or abs(number) > 1.:
            raise ValueError('-1~1 m 범위의 유한한 값을 입력하세요.')
    elif name == 'obstacle_min_height_m':
        if not math.isfinite(number) or number < 0.:
            raise ValueError('0 이상의 유한한 값을 입력하세요.')
    elif not math.isfinite(number) or number <= 0.:
        raise ValueError('0보다 큰 유한한 값을 입력하세요.')
    limits = {'ground_plane_tolerance_m': .08, 'imu_max_skew_s': .2,
              'ground_max_slope_deg': 30., 'avoidance_brake_limit': 1.,
              'lane_change_max_heading_deg': 45., 'lane_change_cancel_max_progress': 1.}
    if name in limits and number > limits[name]:
        raise ValueError('최대 {} {}까지 입력하세요.'.format(limits[name], SETTINGS[name][3]))
    return number


class FrozenConfigTarget:
    """RuntimeSettings target for a frozen dataclass config held by owner.

    Assigning a field replaces owner.<attribute> with an updated copy, so the
    planner always reads one consistent, validated config.
    """
    def __init__(self, owner, attribute='config'):
        object.__setattr__(self, '_owner', owner)
        object.__setattr__(self, '_attribute', attribute)

    def __getattr__(self, name):
        return getattr(getattr(self._owner, self._attribute), name)

    def __setattr__(self, name, value):
        from dataclasses import replace
        config = replace(getattr(self._owner, self._attribute), **{name: value})
        validate = getattr(config, 'validate', None)
        if validate is not None:
            validate()
        setattr(self._owner, self._attribute, config)


class RuntimeSettings:
    """Apply settings on the owning processing loop, then report actual values.

    One namespace read every 0.2 seconds also supports ordinary rosparam edits.
    No callback mutates a planner in the middle of a decision.
    """
    def __init__(self, owner, target):
        self.owner, self.target = owner, target
        self.settings = {key: spec for key, spec in SETTINGS.items() if spec[0] == owner}
        self.next_poll = 0.
        self.publisher = rospy.Publisher('/waypoint_debug/avoidance_config/' + owner,
                                         String, queue_size=1, latch=True)
        self.persistence_file = PERSISTENCE_DIR / ('avoidance_settings_' + owner + '.json')
        saved = self._load_saved()
        for name, spec in self.settings.items():
            value = validated_value(name, saved.get(name, rospy.get_param(
                '~' + name, getattr(target, spec[2], spec[1]))))
            setattr(target, spec[2], value)
            rospy.set_param('~' + name, value)
        if owner == 'lidar_obstacles' and target.cluster_min_size_m > target.cluster_max_size_m:
            raise ValueError('군집 최소 크기는 최대 크기 이하여야 합니다.')
        if owner == 'lidar_obstacles' and target.obstacle_min_height_m >= target.obstacle_max_height_m:
            raise ValueError('장애물 높이 하한은 상한보다 작아야 합니다.')
        self.saved_values = {name: getattr(target, spec[2]) for name, spec in self.settings.items()}

    def _load_saved(self):
        try:
            with self.persistence_file.open(encoding='utf-8') as stream:
                data = json.load(stream)
            if not isinstance(data, dict):
                raise ValueError('설정 파일은 JSON 객체여야 합니다.')
            return {name: validated_value(name, data[name]) for name in self.settings if name in data}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError) as error:
            rospy.logwarn('Saved avoidance settings ignored (%s): %s', self.persistence_file, error)
            return {}

    def _save(self, values):
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=str(self.persistence_file.parent),
                                             prefix='.avoidance_settings_', suffix='.tmp',
                                             delete=False) as stream:
                temporary = stream.name
                json.dump(values, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.persistence_file)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def poll(self, extra=None):
        now = time.monotonic()
        if now < self.next_poll:
            return
        self.next_poll = now + .2
        # Read the master directly: rospy's namespace cache can contain only
        # values set by this node and miss external rosparam/panel updates.
        try:
            code, message, params = rospy.get_master().getParam('~')
            if code != 1 or not isinstance(params, dict):
                raise rospy.ROSException(message)
        except (OSError, rospy.ROSException) as error:
            rospy.logwarn_throttle(5., 'Could not refresh avoidance settings: %s', error)
            params = {}
        errors = {}
        previous_sizes = ((self.target.cluster_min_size_m, self.target.cluster_max_size_m)
                          if self.owner == 'lidar_obstacles' else None)
        previous_heights = ((self.target.obstacle_min_height_m, self.target.obstacle_max_height_m)
                            if self.owner == 'lidar_obstacles' else None)
        for name, spec in self.settings.items():
            previous = getattr(self.target, spec[2])
            try:
                value = validated_value(name, params.get(name, previous))
            except ValueError as error:
                errors[name] = str(error)
                value = previous
            setattr(self.target, spec[2], value)
        if previous_sizes and self.target.cluster_min_size_m > self.target.cluster_max_size_m:
            self.target.cluster_min_size_m, self.target.cluster_max_size_m = previous_sizes
            errors['cluster_min_size_m'] = errors['cluster_max_size_m'] = '최소 크기는 최대 크기 이하여야 합니다.'
        if previous_heights and self.target.obstacle_min_height_m >= self.target.obstacle_max_height_m:
            self.target.obstacle_min_height_m, self.target.obstacle_max_height_m = previous_heights
            errors['obstacle_min_height_m'] = errors['obstacle_max_height_m'] = '높이 하한은 상한보다 작아야 합니다.'
        values = {name: getattr(self.target, spec[2]) for name, spec in self.settings.items()}
        if values != self.saved_values:
            try:
                self._save(values)
                self.saved_values = values.copy()
            except OSError as error:
                rospy.logwarn_throttle(5., 'Avoidance settings could not be saved: %s', error)
                for name, value in values.items():
                    if value != self.saved_values[name]:
                        errors[name] = '적용됐지만 저장 실패: ' + str(error)
        values.update(extra or {})
        self.publisher.publish(String(data=json.dumps(dict(
            node=rospy.get_name(), values=values, errors=errors))))
