"""Relative Ego speed/heading odometry, independent of GPS position."""
import math

from .scan_odometry import wrap


class DeadReckoner:
    def __init__(self, max_interval_s=.25):
        self.max_interval_s = float(max_interval_s)
        if not math.isfinite(self.max_interval_s) or self.max_interval_s <= 0:
            raise ValueError('invalid odometry interval')
        self.pose = None
        self.last_stamp = None

    def update(self, speed_mps, yaw_rad, stamp_s):
        if not all(math.isfinite(v) for v in (speed_mps, yaw_rad, stamp_s)) or speed_mps < 0:
            raise ValueError('invalid Ego speed/heading')
        if self.last_stamp is None:
            self.pose = (0., 0., wrap(yaw_rad))
            self.last_stamp = stamp_s
            return self.pose
        dt = stamp_s - self.last_stamp
        if dt <= 0:
            raise ValueError('stale, frozen or jumped Ego heading')
        if dt > self.max_interval_s:
            # A dropped heading sample must not reset the odom coordinate
            # frame. The caller withholds this sample; the next one resumes
            # from the same pose without integrating an unobserved interval.
            self.last_stamp = stamp_s
            raise ValueError('stale, frozen or jumped Ego heading')
        x, y, old_yaw = self.pose
        middle_yaw = old_yaw + wrap(yaw_rad - old_yaw) / 2.
        self.pose = (x + speed_mps * dt * math.cos(middle_yaw),
                     y + speed_mps * dt * math.sin(middle_yaw),
                     wrap(yaw_rad))
        self.last_stamp = stamp_s
        return self.pose
