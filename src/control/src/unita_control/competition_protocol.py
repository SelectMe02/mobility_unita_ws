"""Competition UDP: official 24.R2.0 AutoMode command and Ego status fields."""
from dataclasses import dataclass
import math
import struct

from .udp_protocol import CommandEncoder, decode_status_mode


class CompetitionCommandEncoder(CommandEncoder):
    """Keep existing validation/scaling, explicitly request UDP AutoMode=2."""
    def pack(self, *arguments):
        packet = bytearray(super().pack(*arguments))
        packet[30] = 2
        return bytes(packet)

    def release(self):
        packet = bytearray(self.stop())
        packet[30] = 1  # Keyboard mode, without a ROS Event service.
        return bytes(packet)


@dataclass(frozen=True)
class EgoStatus:
    mode: int
    gear: int
    stamp: tuple
    speed_kmh: float
    position: tuple
    yaw_deg: float
    wheelbase: float
    steering_deg: float


def decode_ego_status(packet):
    mode, stamp = decode_status_mode(packet)
    speed = struct.unpack_from('<f', packet, 37)[0]
    wheelbase = struct.unpack_from('<f', packet, 69)[0]
    position = struct.unpack_from('<3f', packet, 77)
    yaw = struct.unpack_from('<f', packet, 97)[0]
    steering = struct.unpack_from('<f', packet, 137)[0]
    if not all(math.isfinite(value) for value in (speed, wheelbase, *position, yaw, steering)):
        raise ValueError('non-finite Ego Vehicle Status fields')
    return EgoStatus(mode, packet[36], stamp, speed, position, yaw, wheelbase, steering)


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
