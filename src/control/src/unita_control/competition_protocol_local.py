"""Competition UDP: official 24.R2.0 AutoMode command and Ego status fields."""
from dataclasses import dataclass
import math
import struct

from .udp_protocol import CommandEncoder, decode_status_mode


class CompetitionCommandEncoder(CommandEncoder):
    """Keep existing validation/scaling, explicitly request UDP AutoMode=2."""
    def __init__(self, *args, release_to_keyboard=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.release_to_keyboard = release_to_keyboard

    def encode(self, cmd_type, velocity, acceleration, accel, brake, steering):
        # Regulation v1.2 permits pedal control only on the competition link.
        if cmd_type != 1:
            raise ValueError('competition requires longlCmdType=1 (accel/brake)')
        return super().encode(cmd_type, velocity, acceleration, accel, brake, steering)

    def pack(self, *arguments):
        packet = bytearray(super().pack(*arguments))
        packet[30] = 2
        return bytes(packet)

    def release(self):
        packet = bytearray(self.stop())
        packet[30] = 1 if self.release_to_keyboard else 2
        return bytes(packet)


@dataclass(frozen=True)
class EgoStatus:
    mode: int
    gear: int
    stamp: tuple
    speed_kmh: float
    yaw_deg: float
    wheelbase: float
    steering_deg: float


def decode_ego_status(packet):
    if packet[:11] != b'#MoraiInfo$':
        raise ValueError('expected #MoraiInfo$ Ego status, got {!r}'.format(packet[:16]))
    mode, stamp = decode_status_mode(packet)
    speed = struct.unpack_from('<f', packet, 37)[0]
    wheelbase = struct.unpack_from('<f', packet, 69)[0]
    yaw = struct.unpack_from('<f', packet, 97)[0]
    steering = struct.unpack_from('<f', packet, 137)[0]
    if not all(math.isfinite(value) for value in (speed, wheelbase, yaw, steering)):
        raise ValueError('non-finite Ego Vehicle Status fields')
    # pos_x/y/z (offset 77) is not part of the Competition Vehicle Status
    # regulation and is never read.
    return EgoStatus(mode, packet[36], stamp, speed, yaw, wheelbase, steering)


class StatusFreshness:
    """Repeated frozen SIM packets must not keep authorizing driving."""
    def __init__(self, timeout=1.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('status_timeout must be positive and finite')
        self.timeout = timeout
        self.stamp = None
        self.received = None

    def update(self, stamp, received):
        if stamp == self.stamp:
            return False
        self.stamp, self.received = stamp, received
        return True

    def fresh(self, now):
        return self.received is not None and 0 <= now - self.received <= self.timeout
