"""MORAI EgoCtrlCmd, packed little-endian, 55 bytes with a 14-byte header.

Reference: MORAI-NetworkModule/24.R2.0/lib/define/EgoCtrlCmd.py.
ROS steering is radians; UDP steering is normalized by maximum steering angle.
"""
import math
import struct

PACKET = struct.Struct('<14s4i3b5f2s')

# Installed MORAI S4.251001.MolitComp03 leaves the current control mode
# unchanged for 0. Sending 2 selects external control again on every packet,
# which can undo a Q press before the mode polling thread sees manual mode.
KEEP_CONTROL_MODE = 0


def normalize_service_mode(mode):
    """EventInfo uses 3 for external control; status UDP uses 2."""
    return {1: 1, 3: 2}.get(mode, 0)


def decode_status_mode(packet):
    """Read CtrlMode from 23.R1+ Ego Vehicle Status (base or tire extension)."""
    if (len(packet) not in (181, 229) or packet[:11] != b'#MoraiInfo$'
            or packet[-2:] != b'\r\n'
            or struct.unpack_from('<i', packet, 11)[0] != len(packet) - 29):
        raise ValueError('invalid Ego Vehicle Status UDP frame')
    sec, nsec = struct.unpack_from('<ii', packet, 27)
    if sec < 0 or not 0 <= nsec < 1000000000 or packet[36] > 5:
        raise ValueError('invalid Ego Vehicle Status timestamp/gear')
    return packet[35], (sec, nsec)


class ModeGate:
    """No commands, including brakes, unless fresh SIM status permits control."""
    def __init__(self, timeout=.3):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('status_timeout must be positive and finite')
        self.timeout = timeout
        self.mode = None
        self.received = None
        self.stamp = None

    def update(self, mode, stamp, received):
        # A repeated frozen simulation sample must not keep enabling transport.
        changed = mode != self.mode
        if stamp != self.stamp or changed:
            self.received = received
        self.mode, self.stamp = mode, stamp
        return changed

    def select(self, now):
        if self.received is None or not 0 <= now - self.received <= self.timeout:
            return False, 'PAUSED: missing or stale SIM mode status'
        if self.mode == 1:
            return False, 'PAUSED: SIM manual mode; no control packets'
        if self.mode != 2:
            return False, 'PAUSED: SIM mode %s is not external control' % self.mode
        return True, 'external control'


class CommandEncoder:
    def __init__(self, gear=4, max_steering_deg=36.25, steering_sign=1.0,
                 max_speed_kmh=30.0):
        if gear not in range(6):
            raise ValueError('gear must be 0..5 (D=4)')
        if not math.isfinite(max_steering_deg) or max_steering_deg <= 0:
            raise ValueError('max_steering_deg must be finite and positive')
        if steering_sign not in (-1.0, 1.0):
            raise ValueError('steering_sign must be +1 or -1')
        if not math.isfinite(max_speed_kmh) or not 0 < max_speed_kmh <= 300:
            raise ValueError('max_speed_kmh must be in (0, 300]')
        self.gear = gear
        self.max_steering_rad = math.radians(max_steering_deg)
        self.steering_sign = steering_sign
        self.max_speed = max_speed_kmh

    def pack(self, cmd_type, velocity, acceleration, accel, brake, steer):
        return PACKET.pack(b'#MoraiCtrlCmd$', 23, 0, 0, 0,
                           KEEP_CONTROL_MODE, self.gear, cmd_type,
                           velocity, acceleration, accel, brake, steer, b'\r\n')

    def stop(self):
        # Throttle control with full brake: do not rely on velocity-mode coast.
        return self.pack(1, 0, 0, 0, 1, 0)

    def encode(self, cmd_type, velocity, acceleration, accel, brake, steering):
        if cmd_type not in (1, 2, 3):
            raise ValueError('unsupported longlCmdType')
        if not all(math.isfinite(v) for v in (velocity, acceleration, accel, brake, steering)):
            raise ValueError('command contains non-finite values')
        if not 0 <= accel <= 1 or not 0 <= brake <= 1:
            raise ValueError('accel/brake must be 0..1')
        if abs(acceleration) > 100:
            raise ValueError('acceleration magnitude exceeds 100 m/s^2')
        if velocity < 0:
            raise ValueError('velocity must be nonnegative; reverse requires gear=2')
        if cmd_type == 2 and velocity == 0:
            return self.stop()
        steer = max(-1.0, min(1.0, self.steering_sign * steering / self.max_steering_rad))
        return self.pack(cmd_type, min(velocity, self.max_speed), acceleration, accel, brake, steer)


class CommandWatchdog:
    """Fresh command AND fresh valid GPS/IMU are required for transport."""
    def __init__(self, command_timeout=.5, sensor_timeout=1.0):
        if not all(math.isfinite(v) and v > 0 for v in (command_timeout, sensor_timeout)):
            raise ValueError('timeouts must be positive and finite')
        self.command_timeout = command_timeout
        self.sensor_timeout = sensor_timeout
        self.command = None
        self.sensors = {}

    def set_command(self, packet, received):
        self.command = (packet, received) if packet is not None else None

    def set_sensor(self, name, valid, stamp, received):
        self.sensors[name] = (stamp, received) if valid else None

    def select(self, wall_now, ros_now, stop):
        if self.command is None or not 0 <= wall_now-self.command[1] <= self.command_timeout:
            return stop, 'BRAKE: no fresh /ctrl_cmd'
        for name in ('gps', 'imu'):
            sample = self.sensors.get(name)
            if sample is None:
                return stop, 'BRAKE: invalid or missing ' + name
            stamp, received = sample
            if (not 0 <= wall_now-received <= self.sensor_timeout
                    or not 0 <= ros_now-stamp <= self.sensor_timeout):
                return stop, 'BRAKE: stale or future ' + name
        return self.command[0], 'SENDING: fresh command and sensors (no UDP acknowledgement)'
