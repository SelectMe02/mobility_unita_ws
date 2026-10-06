"""Single-controller GPS/ICP-odometry transport gate.

The command remains the Leo signal guard's /ctrl_cmd in both states. Only
the pose source changes; this gate allows the same command through during a
bounded GPS blackout when the LiDAR odometry state machine is healthy.
"""
import math

from unita_control.udp_protocol import CommandWatchdog


class BlackoutWatchdog(CommandWatchdog):
    def __init__(self, command_timeout=.5, sensor_timeout=1.,
                 fallback_timeout=.35, max_blackout_distance_m=20.,
                 stopline_margin_m=10., max_entry_speed_mps=2.,
                 gps_silence_timeout_s=.75):
        super().__init__(command_timeout, sensor_timeout)
        values = (fallback_timeout, max_blackout_distance_m,
                  stopline_margin_m, max_entry_speed_mps,
                  gps_silence_timeout_s)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('invalid GPS blackout transport limits')
        self.fallback_timeout = fallback_timeout
        self.max_blackout_distance_m = max_blackout_distance_m
        self.stopline_margin_m = stopline_margin_m
        self.max_entry_speed_mps = max_entry_speed_mps
        self.gps_silence_timeout_s = gps_silence_timeout_s
        self.lidar = None
        self.odom = None
        self.speed = None
        self.drive_mode = None
        self.drive_valid = None
        self.avoidance_valid = None
        self.raw_gps = None
        self.guard = None
        self.armed = False
        self.travelled_m = 0.
        self.distance_budget_m = 0.
        self.last_select_wall = None
        self.blackout_started = False

    @staticmethod
    def _fresh(sample, wall_now, ros_now, timeout):
        if sample is None:
            return False
        stamp, received = sample
        return (0 <= wall_now - received <= timeout
                and 0 <= ros_now - stamp <= timeout)

    def _recent(self, sample, now):
        return sample is not None and 0 <= now - sample[-1] <= self.fallback_timeout

    def set_speed(self, speed_mps, received):
        self.speed = (float(speed_mps), received)

    def set_lidar(self, valid, stamp, received):
        self.lidar = (stamp, received) if valid else None

    def set_odom(self, valid, stamp, received):
        self.odom = (stamp, received) if valid else None

    def set_drive_mode(self, mode, received):
        self.drive_mode = (mode, received)

    def set_drive_valid(self, valid, received):
        self.drive_valid = (bool(valid), received)

    def set_avoidance_valid(self, valid, received):
        self.avoidance_valid = (bool(valid), received)

    def set_raw_gps(self, latitude_deg, longitude_deg, status, stamp, received):
        finite = (math.isfinite(latitude_deg) and math.isfinite(longitude_deg)
                  and -80 <= latitude_deg <= 84 and -180 <= longitude_deg <= 180)
        kind = ('zero' if finite and latitude_deg == longitude_deg == 0.0 else
                'valid' if finite and status > 0 else 'invalid')
        self.raw_gps = (kind, stamp, received)

    def _gps_blackout_trigger(self, wall_now, ros_now):
        if self.raw_gps is None:
            return False
        kind, stamp, received = self.raw_gps
        if kind == 'zero' and self._fresh((stamp, received), wall_now, ros_now,
                                          self.fallback_timeout):
            return True
        return (kind in ('valid', 'zero')
                and wall_now - received >= self.gps_silence_timeout_s)

    def set_guard(self, armed, distance_m, received):
        self.guard = (bool(armed), float(distance_m), received)

    def disarm(self):
        self.armed = False
        self.travelled_m = 0.
        self.distance_budget_m = 0.
        self.last_select_wall = None
        self.blackout_started = False

    def _ready_for_fallback(self, wall_now, ros_now, require_slow=True):
        if (not self._fresh(self.lidar, wall_now, ros_now, self.fallback_timeout)
                or not self._fresh(self.odom, wall_now, ros_now, self.fallback_timeout)
                or not self._fresh(self.sensors.get('imu'), wall_now, ros_now,
                                   self.sensor_timeout)):
            return False
        if (not self._recent(self.speed, wall_now) or not math.isfinite(self.speed[0])
                or self.speed[0] < 0
                or (require_slow and self.speed[0] > self.max_entry_speed_mps)):
            return False
        if (not self._recent(self.guard, wall_now) or not self.guard[0]
                or not math.isfinite(self.guard[1])
                or self.guard[1] <= self.stopline_margin_m):
            return False
        return True

    def select(self, wall_now, ros_now, stop):
        normal_packet, normal_reason = super().select(wall_now, ros_now, stop)
        blackout_trigger = self._gps_blackout_trigger(wall_now, ros_now)
        blackout_mode = (self._recent(self.drive_mode, wall_now)
                         and self.drive_mode[0] in ('BLACKOUT', 'SIM_EXIT'))
        if normal_reason.startswith('SENDING:') and not (blackout_trigger and blackout_mode):
            # Recalculate the distance available from the current GPS pose.
            # Travel on the approach must not consume the blackout allowance.
            if (self._ready_for_fallback(wall_now, ros_now, require_slow=False)
                    and self._recent(self.drive_mode, wall_now)
                    and self.drive_mode[0] == 'GPS'):
                self.distance_budget_m = min(
                    self.max_blackout_distance_m,
                    self.guard[1] - self.stopline_margin_m)
                if self.distance_budget_m > 0:
                    self.armed = True
                    self.travelled_m = 0.
                    self.last_select_wall = wall_now
                    self.blackout_started = False
                else:
                    self.disarm()
            else:
                self.disarm()
            return normal_packet, normal_reason
        if self.armed:
            dt = wall_now - self.last_select_wall
            if (not 0 <= dt <= self.sensor_timeout
                    or not self._recent(self.speed, wall_now)
                    or not math.isfinite(self.speed[0]) or self.speed[0] < 0):
                self.disarm()
            else:
                if blackout_trigger:
                    self.travelled_m += self.speed[0] * dt
                self.last_select_wall = wall_now
                if self.travelled_m >= self.distance_budget_m:
                    self.disarm()
        if (not self.armed or (not normal_reason.startswith('SENDING:')
                              and 'gps' not in normal_reason)):
            return stop, normal_reason
        if not blackout_trigger:
            return stop, 'BRAKE: GPS blackout trigger missing'
        if (not self._recent(self.drive_mode, wall_now)
                or self.drive_mode[0] not in ('BLACKOUT', 'SIM_EXIT')
                or not self._recent(self.drive_valid, wall_now)
                or not self.drive_valid[0]):
            return stop, 'BRAKE: LiDAR odometry state not valid'
        if not self._ready_for_fallback(wall_now, ros_now,
                                        require_slow=not self.blackout_started):
            return stop, 'BRAKE: LiDAR, odometry, speed or signal guard unavailable'
        if (not self._recent(self.avoidance_valid, wall_now)
                or not self.avoidance_valid[0]):
            return stop, 'BRAKE: LiDAR obstacle avoidance has no safe gap'
        if self.command is None or not 0 <= wall_now-self.command[1] <= self.command_timeout:
            return stop, 'BRAKE: shared controller command stale'
        self.blackout_started = True
        source = ('MORAI ENU exit pose' if self.drive_mode[0] == 'SIM_EXIT'
                  else 'LiDAR odometry')
        return self.command[0], 'SENDING: shared PP/PID with %s (%.1f/%.1f m)' % (
            source, self.travelled_m, self.distance_budget_m)
