"""Stop release on fresh observations, with immediate brake and ramped throttle."""
import math


class CruiseBrake:
    """Shape ordinary speed tracking only; callers must bypass for hazards."""
    def __init__(self, gain=.025, max_brake=.25, rise_rate=.5, deadband_kmh=1.):
        values = (gain, max_brake, rise_rate, deadband_kmh)
        if (not all(math.isfinite(v) for v in values)
                or min(gain, max_brake, rise_rate) <= 0.
                or max_brake > 1. or deadband_kmh < 0.):
            raise ValueError('invalid cruise brake parameters')
        self.gain, self.max_brake = gain, max_brake
        self.rise_rate, self.deadband_kmh = rise_rate, deadband_kmh
        self.brake = 0.

    def reset(self):
        self.brake = 0.

    def shape(self, output, current_kmh, target_kmh, dt):
        overspeed = max(0., current_kmh-target_kmh-self.deadband_kmh)
        requested = min(max(0., -output), self.max_brake, self.gain*overspeed)
        # Release immediately near the target; never ramp a lingering brake
        # through the requested speed. Only brake application is rate limited.
        self.brake = min(requested, self.brake+self.rise_rate*max(0., min(.2, dt)))
        return -self.brake if output < 0. else output


class PedalGuard:
    def __init__(self):
        self.holding = False
        self.clear_since = None
        self.last_sample = None
        self.clear_samples = 0
        self.accel = 0.
        self.last_time = None

    def stop(self, now):
        self.holding = True
        self.clear_since = None
        self.last_sample = None
        self.clear_samples = 0
        self.accel = 0.
        self.last_time = now

    def command(self, output, now, sample_stamp=None, hold_s=.4,
                accel_rate=.5, brake_limit=1.):
        dt = .05 if self.last_time is None else max(0., min(.2, now-self.last_time))
        self.last_time = now
        if self.holding:
            if self.last_sample is not None and sample_stamp is not None and sample_stamp < self.last_sample:
                self.clear_since, self.clear_samples = None, 0
            if self.clear_since is None:
                self.clear_since = now
            if sample_stamp is None or sample_stamp != self.last_sample:
                self.clear_samples += 1
            self.last_sample = sample_stamp
            if now-self.clear_since < hold_s or self.clear_samples < 3:
                return 0., 1., 'clearance_confirmation'
            self.holding = False
        if output <= 0.:
            self.accel = 0.
            return 0., min(brake_limit, max(0., -output)), 'brake' if output < 0. else 'coast'
        self.accel = min(output, self.accel+accel_rate*dt)
        return self.accel, 0., 'drive'
