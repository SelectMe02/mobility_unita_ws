"""ROS-independent checks for the one-shot MORAI field diagnostic."""
from dataclasses import dataclass, field
import ipaddress
import math


@dataclass
class Check:
    name: str
    status: str
    detail: str = ''
    causes: tuple = ()


def aggregate(checks):
    statuses = {check.status for check in checks}
    if 'FAIL' in statuses:
        return 'NOT READY', 1
    if 'WARN' in statuses:
        return 'READY WITH WARNINGS', 0
    return 'READY', 0


def ip_check(address, field_mode=True):
    try:
        ip = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        return Check('Client IPv4', 'FAIL', 'Expected the Windows Client PC IPv4')
    if ip.is_unspecified or ip.is_multicast or str(ip) == '255.255.255.255':
        return Check('Client IPv4', 'FAIL', 'Expected a unicast Windows Client PC IPv4')
    if ip.is_loopback and field_mode:
        return Check('Client IPv4', 'WARN', 'Loopback cannot reach a different Windows PC',
                     ('Set morai_ip to the Windows Ethernet IPv4.',))
    return Check('Client IPv4', 'PASS', str(ip))


def measured_hz(times):
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return (len(times) - 1) / (times[-1] - times[0])


def stamp_check(stamp, now, timeout):
    if not math.isfinite(stamp) or not math.isfinite(now) or stamp < 0:
        return 'FAIL', 'invalid timestamp'
    if stamp == 0:
        return 'WARN', 'timestamp=0; source freshness cannot be verified'
    age = now - stamp
    if age < -0.1:
        return 'FAIL', 'future timestamp, age={:.3f}s'.format(age)
    if age > timeout:
        return 'FAIL', 'stale timestamp, age={:.3f}s'.format(age)
    return 'PASS', 'timestamp age={:.3f}s'.format(age)


def sensor_detail(kind, message):
    """Validate already decoded ROS fields; does not implement UDP protocols."""
    if kind == 'camera':
        if message.width <= 0 or message.height <= 0 or not message.data:
            raise ValueError('empty image or invalid dimensions')
        if message.step <= 0 or len(message.data) < message.step * message.height:
            raise ValueError('image data is shorter than step * height')
        return '{}x{}, encoding={}'.format(message.width, message.height, message.encoding)
    if kind == 'velodyne':
        if message.width <= 0 or message.height <= 0 or not message.data:
            raise ValueError('empty point cloud')
        if message.point_step <= 0 or len(message.data) < message.point_step * message.width * message.height:
            raise ValueError('point cloud data is incomplete')
        return '{} points'.format(message.width * message.height)
    if kind == 'gps':
        lat, lon = message.latitude, message.longitude
        if not all(math.isfinite(value) for value in (lat, lon)):
            raise ValueError('non-finite GPS coordinates')
        if not -80 <= lat <= 84 or not -180 <= lon <= 180:
            raise ValueError('GPS coordinates outside the localization UTM bounds')
        if lat == lon == 0:
            raise ValueError('GPS coordinate is 0,0')
        if message.status <= 0:
            raise ValueError('GPS has no fix (status <= 0)')
        return 'lat={:.8f}, lon={:.8f}, status={}'.format(lat, lon, message.status)
    if kind == 'imu':
        q = message.orientation
        values = (q.x, q.y, q.z, q.w)
        norm = math.sqrt(sum(value * value for value in values))
        if not all(math.isfinite(value) for value in values) or not math.isfinite(norm) or norm < 1e-6:
            raise ValueError('non-finite or zero IMU quaternion')
        if message.orientation_covariance[0] == -1:
            raise ValueError('IMU orientation is unavailable')
        return 'quaternion norm={:.6f}'.format(norm)
    if kind == 'control':
        if not all(math.isfinite(value) for value in (message.velocity, message.steering)):
            raise ValueError('non-finite CtrlCmd velocity/steering')
        if message.longlCmdType not in (1, 2, 3):
            raise ValueError('invalid CtrlCmd longlCmdType')
        return 'velocity={:.1f}km/h, steering={:.3f}rad'.format(message.velocity, message.steering)
    if kind == 'mode':
        return {1: 'Manual', 2: 'ExternalCtrl', 0: 'Built-in / other / unknown'}.get(message.data, 'Unknown')
    if kind == 'udp_status':
        return message.data
    raise ValueError('unsupported diagnostic kind: ' + kind)


@dataclass
class SensorStats:
    name: str
    kind: str
    required: bool = True
    times: list = field(default_factory=list)
    stamps: set = field(default_factory=set)
    errors: list = field(default_factory=list)
    last: object = None
    detail: str = ''

    def observe(self, message, received):
        self.times.append(received)
        self.last = message
        if hasattr(message, 'header'):
            self.stamps.add(message.header.stamp.to_sec())
        try:
            self.detail = sensor_detail(self.kind, message)
        except (ValueError, AttributeError, TypeError, OverflowError) as error:
            self.errors.append(str(error))

    def result(self, wall_now, ros_now, timeout, causes=()):
        level = 'FAIL' if self.required else 'WARN'
        if not self.times:
            return Check(self.name, level, 'No ROS messages received during the measurement window', causes)
        hz = measured_hz(self.times)
        rate = '{:.1f} Hz'.format(hz) if hz is not None else 'Hz unavailable (one message)'
        detail = '{}; n={}; {}'.format(rate, len(self.times), self.detail)
        if self.errors:
            return Check(self.name, level, detail + '; invalid samples: ' + self.errors[-1], causes)
        if wall_now - self.times[-1] > timeout:
            return Check(self.name, level, detail + '; reception stopped', causes)
        status = 'PASS'
        if hasattr(self.last, 'header'):
            detail += '; frame_id=' + self.last.header.frame_id
            state, age = stamp_check(self.last.header.stamp.to_sec(), ros_now, timeout)
            detail += '; ' + age
            if state == 'FAIL':
                return Check(self.name, level, detail, causes)
            if state == 'WARN':
                status = 'WARN'
            elif len(self.times) > 1 and len(self.stamps) == 1:
                return Check(self.name, level, detail + '; source timestamp is not advancing', causes)
        return Check(self.name, status, detail)
