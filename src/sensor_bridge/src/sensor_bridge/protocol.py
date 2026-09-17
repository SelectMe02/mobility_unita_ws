"""Packed little-endian formats: MORAI-NetworkModule 24.R2.0 / sensor manual."""
import math
import re
import struct
import time


def valid_stamp(sec, nsec):
    if sec < 0 or not 0 <= nsec < 1000000000:
        raise ValueError('invalid sensor timestamp')
    return sec, nsec


def parse_imu(packet):
    # NetworkModule has a timestamp; the 24.R2.2 manual omits it.
    if len(packet) not in (107, 115) or packet[:9] != b'#IMUData$':
        raise ValueError('expected 107/115-byte #IMUData$ packet')
    if packet[-2:] != b'\r\n':
        raise ValueError('invalid IMU tail')
    size = struct.unpack_from('<I', packet, 9)[0]
    if size not in ((80,) if len(packet) == 107 else (80, 88)):
        raise ValueError('invalid IMU payload size')
    stamp = None
    offset = 25
    if len(packet) == 115:
        stamp = valid_stamp(*struct.unpack_from('<ii', packet, offset))
        offset += 8
    values = struct.unpack_from('<10d', packet, offset)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('non-finite IMU value')
    if sum(v * v for v in values[:4]) < 1e-12:
        raise ValueError('zero IMU quaternion')
    return stamp, values


class CameraAssembler:
    """Drop incomplete/out-of-order frames instead of mixing JPEG fragments."""

    def __init__(self, max_bytes=16 * 1024 * 1024, timeout=1.0):
        self.max_bytes = max_bytes
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.stamp = None
        self.next_index = None
        self.buffer = bytearray()
        self.started = 0.0

    def feed(self, packet):
        if packet[:3] == b'BOX':
            return None
        if len(packet) < 21 or packet[:3] != b'MOR':
            self.reset()
            raise ValueError('expected timestamped MOR camera packet')
        sec, nsec, index, size = struct.unpack_from('<4i', packet, 3)
        stamp = valid_stamp(sec, nsec)
        if not 0 < size <= 64979 or len(packet) not in (size + 21, 65000):
            self.reset()
            raise ValueError('invalid camera fragment size')
        tail = packet[-2:]
        if tail not in (b'AI', b'EI'):
            self.reset()
            raise ValueError('invalid camera tail')
        chunk = packet[19:19 + size]
        now = time.monotonic()
        if chunk.startswith(b'\xff\xd8'):
            self.reset()
            self.stamp = stamp
            self.next_index = index
            self.started = now
        if (self.stamp != stamp or self.next_index != index
                or now - self.started > self.timeout):
            self.reset()
            raise ValueError('missing, reordered or expired camera fragment')
        if len(self.buffer) + size > self.max_bytes:
            self.reset()
            raise ValueError('camera frame exceeds memory limit')
        self.buffer.extend(chunk)
        self.next_index += 1
        if tail == b'EI':
            jpeg = bytes(self.buffer)
            self.reset()
            if not jpeg.endswith(b'\xff\xd9'):
                raise ValueError('incomplete JPEG image')
            return stamp, jpeg
        return None


def _coordinate(value, direction, latitude):
    raw = float(value)
    degrees, minutes = divmod(raw, 100)
    limit = 90 if latitude else 180
    allowed = ('N', 'S') if latitude else ('E', 'W')
    result = degrees + minutes / 60.0
    if (not math.isfinite(raw) or raw < 0 or minutes >= 60
            or result > limit or direction not in allowed):
        raise ValueError('invalid NMEA coordinate')
    return -result if direction in ('S', 'W') else result


def parse_gps(packet):
    """Yield GGA fixes. RMC is redundant and has no altitude/map offsets."""
    text = packet.decode('ascii').rstrip('\x00')
    for sentence in re.findall(r'\$[^$\r\n\x00]+', text):
        try:
            body = sentence[1:]
            if '*' in body:
                body, checksum = body.split('*', 1)
                expected = 0
                for character in body:
                    expected ^= ord(character)
                if len(checksum) != 2 or int(checksum, 16) != expected:
                    continue
            fields = body.split(',')
            if fields[0] not in ('GPGGA', 'GNGGA') or len(fields) < 11:
                continue
            quality = int(fields[6])
            if quality <= 0 or fields[10] != 'M':
                continue
            lat = _coordinate(fields[2], fields[3], True)
            lon = _coordinate(fields[4], fields[5], False)
            altitude = float(fields[9])
            if not math.isfinite(altitude):
                continue
            yield lat, lon, altitude, quality
        except (ValueError, IndexError):
            continue
