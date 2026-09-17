"""ROS-independent planar localization. No inertial integration or EKF."""
import math
from pyproj import Proj
from pyproj.exceptions import ProjError


class GPSIMUEstimator:
    def __init__(self, utm_zone=52, south=False, timeout=1.0,
                 max_skew=0.2, yaw_offset=0.0, require_gps_fix=True):
        if not 1 <= utm_zone <= 60:
            raise ValueError('utm_zone must be between 1 and 60')
        if not all(math.isfinite(v) and v > 0 for v in (timeout, max_skew)):
            raise ValueError('timeout and max_skew must be positive and finite')
        if not math.isfinite(yaw_offset):
            raise ValueError('yaw_offset must be finite')
        self.project = Proj(proj='utm', zone=utm_zone, south=south, ellps='WGS84')
        self.timeout = timeout
        self.max_skew = max_skew
        self.yaw_offset = yaw_offset
        self.require_gps_fix = require_gps_fix
        self.gps = None
        self.imu = None

    def update_gps(self, latitude, longitude, east_offset, north_offset,
                   status, stamp, received):
        self.gps = None
        values = (latitude, longitude, east_offset, north_offset, stamp, received)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('non-finite GPS value')
        if not (-80 <= latitude <= 84 and -180 <= longitude <= 180):
            raise ValueError('GPS coordinates outside UTM bounds')
        if latitude == longitude == 0:
            raise ValueError('GPS zero coordinate (possible signal loss)')
        if self.require_gps_fix and status <= 0:
            raise ValueError('GPS has no valid fix')
        try:
            x, y = self.project(longitude, latitude, errcheck=True)
        except ProjError as error:
            raise ValueError('GPS projection failed: {}'.format(error)) from error
        x, y = x - east_offset, y - north_offset
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError('invalid projected GPS position')
        self.gps = (x, y, stamp, received)

    def update_imu(self, quaternion, orientation_available, stamp, received):
        self.imu = None
        if not orientation_available:
            raise ValueError('IMU orientation unavailable')
        if len(quaternion) != 4 or not all(
                math.isfinite(v) for v in (*quaternion, stamp, received)):
            raise ValueError('invalid IMU quaternion/timestamp')
        norm = math.sqrt(sum(v * v for v in quaternion))
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError('invalid IMU quaternion norm')
        x, y, z, w = (v / norm for v in quaternion)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw += self.yaw_offset
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        self.imu = (yaw, stamp, received)

    def snapshot(self, now, monotonic_now):
        if self.gps is None or self.imu is None:
            return None, 'waiting for valid GPS and IMU'
        for name, sample in (('GPS', self.gps), ('IMU', self.imu)):
            stamp, received = sample[-2:]
            # Wall time catches stalled /clock; ROS time catches stale stamps.
            if not (0 <= now - stamp <= self.timeout
                    and 0 <= monotonic_now - received <= self.timeout):
                return None, name + ' is stale or has a future timestamp'
        if abs(self.gps[-2] - self.imu[-2]) > self.max_skew:
            return None, 'GPS/IMU timestamp difference exceeds max_sensor_skew'
        x, y = self.gps[:2]
        yaw = self.imu[0]
        # Do not make an old measurement appear new on each timer tick.
        stamp = min(self.gps[-2], self.imu[-2])
        return (x, y, yaw, stamp), ''
