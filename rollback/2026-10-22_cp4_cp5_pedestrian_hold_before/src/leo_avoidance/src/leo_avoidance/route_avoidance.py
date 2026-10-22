"""Local path around obstacles on the global route; yield to movers.

Obstacles are expressed in the route frame: s along the route and d to its
left. Ego follows the global route with a lateral offset profile (the local
path). A static block in Ego's lane is passed on an adjacent same-direction
lane only where the map allows a lane change (dashed line) over the whole
pass, and only when that lane is clear from now until the return, including
traffic that would arrive meanwhile. A solid line or an occupied lane means
waiting before the block. A moving obstacle whose predicted track enters
Ego's swept band around the time Ego reaches it caps speed so Ego can stop
before it. No ROS dependency.
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RouteAvoidanceConfig:
    vehicle_half_width_m: float = .946
    side_margin_m: float = .4          # band half width = vehicle + margin + obstacle radius
    front_extent_m: float = 3.845      # rear axle to front bumper
    rear_extent_m: float = .79         # rear axle to rear bumper
    plan_distance_m: float = 40.       # start passes for blocks closer than this
    path_length_m: float = 60.
    path_step_m: float = 1.
    ramp_in_m: float = 14.
    min_ramp_in_m: float = 6.
    ramp_out_m: float = 14.
    pre_gap_m: float = 2.             # front bumper to block when fully offset
    post_gap_m: float = 2.            # rear bumper past block before returning
    merge_gap_m: float = 15.          # pass consecutive blocks closer than this together
    stop_gap_m: float = 5.            # front bumper to block when waiting
    follow_gap_m: float = 6.          # front bumper to a lead mover
    pass_speed_kmh: float = 20.
    brake_mps2: float = 3.
    reaction_s: float = .4
    predict_s: float = 4.             # crossing prediction horizon
    arrival_before_s: float = 1.      # conflict window around Ego arrival
    arrival_after_s: float = 1.5
    moving_speed_mps: float = .5      # route-frame speed that counts as moving
    oncoming_margin_m: float = 5.
    window_behind_m: float = 10.
    lateral_window_m: float = 9.
    oncoming_view_m: float = 40.      # extra look ahead for oncoming traffic
    stop_line_clear_m: float = 5.     # a pass may not end this close past a stop line
    crosswalk_person_m: float = 1.      # half extents up to this look like a person
    crosswalk_stop_gap_m: float = 2.    # front bumper to the crosswalk when stopped

    def band(self, radius_m):
        return self.vehicle_half_width_m + self.side_margin_m + radius_m


@dataclass(frozen=True)
class RouteObstacle:
    """Map-frame obstacle: centre, half extents and velocity.

    radius is the half width across the route; half_length (default radius)
    the half extent along it, so a car is long but not wide.
    """
    x: float
    y: float
    radius: float
    vx: float = 0.
    vy: float = 0.
    dynamic: bool = False
    track_id: int = -1
    half_length: float = None


@dataclass
class Decision:
    state: str
    path: list
    speed_cap_mps: float
    reason: str
    offset_now_m: float = 0.
    obstacles: list = field(default_factory=list)
    # Keep traffic constraints even if a static-path wait already has a
    # lower cap and therefore owns the displayed state/reason.
    traffic_speed_cap_mps: float = math.inf


def stopping_speed(distance_m, brake_mps2, reaction_s):
    """Largest speed that stops within distance_m after the reaction time."""
    distance_m = max(0., distance_m)
    return (math.sqrt((brake_mps2 * reaction_s) ** 2 + 2. * brake_mps2 * distance_m)
            - brake_mps2 * reaction_s)


def smoothstep(t):
    t = max(0., min(1., t))
    return t * t * (3. - 2. * t)


class RouteFrame:
    """Arc length / signed left offset on a polyline route."""

    def __init__(self, points):
        points = np.asarray(points, float)[:, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]
        self.points = points[keep]
        if len(self.points) < 2:
            raise ValueError('route needs at least two distinct points')
        delta = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(delta, axis=1)
        self.tangents = delta / self.lengths[:, None]
        self.s = np.r_[0., np.cumsum(self.lengths)]

    def window(self, s_from, s_to):
        lo = max(0, int(np.searchsorted(self.s, s_from, side='right')) - 1)
        hi = min(len(self.lengths), int(np.searchsorted(self.s, s_to, side='left')) + 1)
        return lo, max(lo + 1, hi)

    def project(self, x, y, lo, hi):
        """(s, d, segment) of (x, y) on segments lo..hi-1."""
        start = self.points[lo:hi]
        tangent = self.tangents[lo:hi]
        rel = np.array((x, y)) - start
        along = np.clip(np.sum(rel * tangent, axis=1), 0., self.lengths[lo:hi])
        foot = start + along[:, None] * tangent
        distance = np.linalg.norm(np.array((x, y)) - foot, axis=1)
        k = int(np.argmin(distance))
        seg = lo + k
        t = self.tangents[seg]
        cross = t[0] * (y - self.points[seg][1]) - t[1] * (x - self.points[seg][0])
        return float(self.s[seg] + along[k]), float(cross), seg

    def project_many(self, xy, lo, hi):
        """(s, d) arrays for many points on segments lo..hi-1 (nearest segment
        start, then its tangent; the route is sampled densely)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        _, k = cKDTree(self.points[lo:hi]).query(xy)
        seg = lo + k
        t = self.tangents[seg]
        off = xy - self.points[seg]
        along = np.clip(np.sum(off * t, axis=1), -self.lengths[seg], self.lengths[seg])
        d = t[:, 0] * off[:, 1] - t[:, 1] * off[:, 0]
        return self.s[seg] + along, d

    def pose(self, s, d=0.):
        seg = min(len(self.lengths) - 1, max(0, int(np.searchsorted(self.s, s, side='right')) - 1))
        t = self.tangents[seg]
        p = self.points[seg] + t * (s - self.s[seg])
        return (float(p[0] - t[1] * d), float(p[1] + t[0] * d)), seg


@dataclass
class _Pass:
    ramp_start_s: float
    full_s: float
    hold_end_s: float
    end_s: float
    offset_m: float
    block_first_s: float

    def offset_at(self, s):
        if s <= self.ramp_start_s or s >= self.end_s:
            return 0.
        if s < self.full_s:
            return self.offset_m * smoothstep(
                (s - self.ramp_start_s) / max(self.full_s - self.ramp_start_s, 1e-6))
        if s <= self.hold_end_s:
            return self.offset_m
        return self.offset_m * (1. - smoothstep(
            (s - self.hold_end_s) / max(self.end_s - self.hold_end_s, 1e-6)))


class RouteAvoidance:
    def __init__(self, route_points, config=None, lane_changes=(), crosswalks=()):
        """lane_changes: [{start_s, end_s, side, offset_m}] where a dashed line
        allows moving to the adjacent lane offset_m to the left (+).
        crosswalks: [(start_s, end_s)] along the route; an obstacle overlapping
        the global route there makes Ego stop before it."""
        self.config = config or RouteAvoidanceConfig()
        cfg = self.config
        points = np.asarray(route_points, float)[:, :2]
        base = RouteFrame(points)
        # A closed lap (last point back at the start) continues past the
        # start line: append the first stretch again so locating Ego, the
        # obstacle window and the output path run on across the seam.
        self.loop_length = None
        lane_changes = [dict(start_s=float(v['start_s']), end_s=float(v['end_s']),
                             side=v['side'], offset_m=float(v['offset_m']))
                        for v in lane_changes]
        crosswalks = [(float(a), float(b)) for a, b in crosswalks]
        if np.linalg.norm(base.points[-1] - base.points[0]) < 1.:
            self.loop_length = float(base.s[-1])
            extension = (cfg.path_length_m + cfg.oncoming_view_m
                         + cfg.window_behind_m + 60.)
            count = int(np.searchsorted(base.s, extension)) + 1
            points = np.vstack((base.points, base.points[1:count + 1]))
            lane_changes += [dict(v, start_s=v['start_s'] + self.loop_length,
                                  end_s=v['end_s'] + self.loop_length)
                             for v in lane_changes if v['start_s'] < extension]
            crosswalks += [(a + self.loop_length, b + self.loop_length)
                           for a, b in crosswalks if a < extension]
        self.route = RouteFrame(points)
        self.crosswalks = sorted(crosswalks)
        self.crosswalk_hold = None  # crosswalk start currently occupied on the route
        self.lane_changes = sorted(lane_changes, key=lambda v: v['start_s'])
        self.active = None
        self.last_s = None

    def reset(self):
        self.active = None
        self.last_s = None
        self.crosswalk_hold = None

    def _locate_ego(self, x, y):
        if self.last_s is None:
            lo, hi = 0, len(self.route.lengths)
        else:
            lo, hi = self.route.window(self.last_s - 15., self.last_s + 40.)
        s, d, _ = self.route.project(x, y, lo, hi)
        if self.loop_length is not None and s >= self.loop_length:
            # Crossed the start line: continue on the first lap stretch.
            s -= self.loop_length
            if self.active is not None:
                shift = self.loop_length
                self.active = replace(
                    self.active, ramp_start_s=self.active.ramp_start_s - shift,
                    full_s=self.active.full_s - shift,
                    hold_end_s=self.active.hold_end_s - shift,
                    end_s=self.active.end_s - shift,
                    block_first_s=self.active.block_first_s - shift)
            if self.crosswalk_hold is not None:
                self.crosswalk_hold -= self.loop_length
        self.last_s = s
        return s, d

    def _to_route(self, obstacles, s_ego):
        cfg = self.config
        reach = cfg.path_length_m + cfg.oncoming_view_m
        lo, hi = self.route.window(s_ego - cfg.window_behind_m - 5., s_ego + reach + 10.)
        result = []
        for o in obstacles:
            s, d, seg = self.route.project(o.x, o.y, lo, hi)
            if (not s_ego - cfg.window_behind_m <= s <= s_ego + reach
                    or abs(d) > cfg.lateral_window_m
                    # Waypoint-only zones (tollgate): fixed structures there
                    # are cleared by the route itself, so nothing is avoided.
                    or any(a <= s <= b for a, b in getattr(self, 'ignored_s_ranges', ()))):
                continue
            t = self.route.tangents[seg]
            v_s = o.vx * t[0] + o.vy * t[1]
            v_d = -o.vx * t[1] + o.vy * t[0]
            result.append(dict(s=s, d=d, r=o.radius,
                               l=o.radius if o.half_length is None else o.half_length,
                               v_s=v_s, v_d=v_d,
                               moving=o.dynamic and math.hypot(v_s, v_d)
                               >= cfg.moving_speed_mps, id=o.track_id))
        return result

    def pass_offsets(self, s_from, s_to):
        """Adjacent-lane offsets whose dashed line covers all of [s_from, s_to]."""
        result = []
        for side in ('left', 'right'):
            covered, offset = s_from, None
            for v in self.lane_changes:
                if v['side'] != side or v['end_s'] < covered or v['start_s'] > covered + 1.:
                    continue
                covered = max(covered, v['end_s'])
                offset = v['offset_m'] if offset is None else offset
                if covered >= s_to:
                    result.append(offset)
                    break
        return result

    def _in_band(self, o, offset):
        return abs(o['d'] - offset) < self.config.band(o['r'])

    def _lane_clear(self, items, offset, s_from, s_to, ego_speed):
        """No obstacle in the band at offset over [s_from, s_to], now or on arrival."""
        cfg = self.config
        duration = (s_to - s_from) / max(ego_speed, 3.)
        for o in items:
            # Blocks in Ego's own lane are what the pass goes around.
            if not self._in_band(o, offset) or abs(o['d']) < abs(o['d'] - offset):
                continue
            s_now = o['s']
            s_then = o['s'] + o['v_s'] * duration
            low, high = min(s_now, s_then) - o['l'], max(s_now, s_then) + o['l']
            if high >= s_from - cfg.oncoming_margin_m and low <= s_to + cfg.oncoming_margin_m:
                return False, o
        return True, None

    def update(self, ego_xy, ego_speed_mps, obstacles, stop_line_distance_m=None,
               stop_line_permitted=True):
        """Return the offset path (map points), a speed cap and the state."""
        cfg = self.config
        speed = max(0., ego_speed_mps if math.isfinite(ego_speed_mps) else 0.)
        s_ego, _ = self._locate_ego(*ego_xy)
        items = self._to_route(obstacles, s_ego)
        front_s = s_ego + cfg.front_extent_m
        state, reason = 'FOLLOW', 'clear'
        cap = math.inf
        traffic_cap = math.inf

        if self.active is not None and s_ego > self.active.end_s:
            self.active = None
        offset_now = self.active.offset_at(s_ego) if self.active else 0.

        # Static blocks in Ego's own lane ahead of the bumper (or beside it while passing).
        blocks = sorted((o for o in items if not o['moving'] and self._in_band(o, 0.)
                         and o['s'] + o['l'] > front_s - (cfg.front_extent_m
                                                           if self.active else 0.)),
                        key=lambda o: o['s'])
        if self.active is not None:
            # A further block soon after the current one extends the hold.
            for o in blocks:
                if o['s'] - o['l'] <= self.active.hold_end_s + cfg.merge_gap_m:
                    self.active.hold_end_s = max(
                        self.active.hold_end_s,
                        o['s'] + o['l'] + cfg.rear_extent_m + cfg.post_gap_m)
                    self.active.end_s = max(self.active.end_s,
                                            self.active.hold_end_s + cfg.ramp_out_m)
            clear, blocker = self._lane_clear(items, self.active.offset_m, s_ego,
                                              self.active.end_s, speed)
            beside = front_s >= self.active.block_first_s - cfg.pre_gap_m
            if not clear and not beside:
                self.active = None
                offset_now = 0.
                reason = 'pass lane blocked at s+%.1f m; return and wait' % (
                    blocker['s'] - s_ego)
            elif not clear:
                gap = blocker['s'] - blocker['l'] - front_s - cfg.stop_gap_m
                cap = min(cap, stopping_speed(gap, cfg.brake_mps2, cfg.reaction_s))
                state, reason = 'PASSING', 'pass lane obstacle ahead; slowing'
            else:
                state, reason = 'PASSING', 'passing static block'
        if self.active is None and blocks and blocks[0]['s'] - s_ego <= cfg.plan_distance_m:
            first = blocks[0]
            last_end = first['s'] + first['l']
            for o in blocks[1:]:
                if o['s'] - o['l'] <= last_end + cfg.merge_gap_m:
                    last_end = max(last_end, o['s'] + o['l'])
            block_first_s = first['s'] - first['l']
            full_s = block_first_s - cfg.pre_gap_m - cfg.front_extent_m
            ramp = min(cfg.ramp_in_m, full_s - s_ego)
            hold_end = last_end + cfg.rear_extent_m + cfg.post_gap_m
            end_s = hold_end + cfg.ramp_out_m
            creep = ramp < cfg.min_ramp_in_m
            ramp_start = s_ego if creep else full_s - ramp
            full = max(s_ego + cfg.min_ramp_in_m, full_s) if creep else full_s
            hold_end, end_s = max(hold_end, full), max(end_s, full + cfg.ramp_out_m)
            offsets = self.pass_offsets(ramp_start, end_s)
            clear, blocker, pass_offset = False, None, None
            for offset in offsets:
                clear, blocker = self._lane_clear(items, offset, s_ego, end_s, speed)
                if clear:
                    pass_offset = offset
                    break
            stop_line_in_pass = (stop_line_distance_m is not None
                                 and not stop_line_permitted
                                 and 0. <= stop_line_distance_m
                                 <= end_s - front_s + cfg.stop_line_clear_m)
            gap = block_first_s - front_s - cfg.stop_gap_m
            side = lambda offset: 'left' if offset > 0 else 'right'
            if clear and not stop_line_in_pass and (not creep or speed < .3):
                self.active = _Pass(ramp_start, full, hold_end, end_s,
                                    pass_offset, block_first_s)
                offset_now = self.active.offset_at(s_ego)
                state = 'PASSING'
                reason = ('pulling out to the %s lane from behind a static block' if creep
                          else 'passing static block on the %s lane') % side(pass_offset)
                if creep:
                    # Waiting close behind it: creep out over the room that is left.
                    cap = min(cap, 7. / 3.6)
            else:
                cap = min(cap, stopping_speed(gap, cfg.brake_mps2, cfg.reaction_s))
                if stop_line_in_pass:
                    traffic_cap = min(traffic_cap, cap)
                if not offsets:
                    state, reason = 'WAIT_SOLID_LINE', 'solid line beside the block; waiting'
                else:
                    state = 'WAIT_PASS'
                    reason = ('red stop line inside the pass' if stop_line_in_pass else
                              '%s lane occupied at s+%.1f m' % (side(offsets[-1]), blocker['s'] - s_ego)
                              if blocker is not None else 'too close to pull out; slowing')
        if self.active is not None:
            cap = min(cap, cfg.pass_speed_kmh / 3.6)

        def offset_at(s):
            return self.active.offset_at(s) if self.active else 0.

        # Anything static left on the planned path must still be stoppable for.
        for o in items:
            if o['moving'] or not self._in_band(o, offset_at(o['s'])):
                continue
            ahead = o['s'] - o['l'] - front_s
            if ahead > -cfg.front_extent_m:
                stop = stopping_speed(ahead - cfg.stop_gap_m, cfg.brake_mps2, cfg.reaction_s)
                if stop < cap:
                    cap = stop
                    if state == 'FOLLOW':
                        state, reason = 'WAIT_PASS', 'static block ahead beyond pass planning'

        # Crosswalk: stop only when the current obstacle footprint overlaps
        # the global route. An object in another lane does not occupy our path.
        for start, end in self.crosswalks:
            ahead = start - front_s
            if ahead < -.5 or ahead > cfg.path_length_m:
                if self.crosswalk_hold == start:
                    self.crosswalk_hold = None
                continue
            occupied = False
            for o in items:
                person = (o['r'] <= cfg.crosswalk_person_m
                          and o['l'] <= cfg.crosswalk_person_m)
                if not (start - o['l'] <= o['s'] <= end + o['l']):
                    continue
                if abs(o['d']) > cfg.band(o['r']):
                    continue
                if (o['moving'] and abs(o['v_s']) >= cfg.moving_speed_mps
                        and abs(o['v_s']) >= 2. * abs(o['v_d'])):
                    continue  # traffic along the lane: trailed or ignored, not a crossing
                if not (person or o['moving']):
                    continue
                occupied = True
                reason_here = 'person on the crosswalk at s+%.1f m, d %.1f m' % (
                    o['s'] - s_ego, o['d'])
            if occupied:
                self.crosswalk_hold = start
            elif self.crosswalk_hold == start:
                self.crosswalk_hold = None
            if self.crosswalk_hold == start:
                stop = stopping_speed(ahead - cfg.crosswalk_stop_gap_m,
                                      cfg.brake_mps2, cfg.reaction_s)
                traffic_cap = min(traffic_cap, stop)
                if stop < cap:
                    cap, state = stop, 'CROSSWALK_STOP'
                    reason = (reason_here if occupied else
                              'crosswalk cleared; completing the stop')

        # Movers: lead traffic in the path and predicted crossings.
        for o in items:
            if not o['moving']:
                continue
            off = offset_at(o['s'])
            ahead = o['s'] - o['l'] - front_s
            if ahead < -cfg.front_extent_m:
                continue
            if self._in_band(o, off):
                if ahead <= 0.:
                    traffic_cap = 0.
                    cap, state, reason = 0., 'YIELD', 'mover beside the bumper'
                    continue
                if o['v_s'] > cfg.moving_speed_mps and abs(o['v_d']) < cfg.moving_speed_mps:
                    # Lead traffic: keep a gap that still stops if it brakes hard.
                    lead = stopping_speed(ahead - cfg.follow_gap_m, cfg.brake_mps2,
                                          cfg.reaction_s) + o['v_s']
                    traffic_cap = min(traffic_cap, lead)
                    if lead < cap:
                        cap, state, reason = lead, 'FOLLOW_LEAD', 'following a lead mover'
                    continue
                stop = stopping_speed(ahead - cfg.stop_gap_m + 2., cfg.brake_mps2, cfg.reaction_s)
                traffic_cap = min(traffic_cap, stop)
                if stop < cap:
                    cap, state, reason = stop, 'YIELD', 'mover in the path'
                continue
            arrival = ahead / max(speed, 1.)
            t0 = max(0., arrival - cfg.arrival_before_s)
            t1 = min(cfg.predict_s, arrival + cfg.arrival_after_s)
            if t1 < t0:
                continue
            band = cfg.band(o['r'])
            hit = False
            for t in np.linspace(t0, t1, 12):
                d = o['d'] + o['v_d'] * t
                if abs(d - offset_at(o['s'] + o['v_s'] * t)) < band:
                    hit = True
                    break
            if hit:
                stop = stopping_speed(ahead - cfg.stop_gap_m + 2., cfg.brake_mps2, cfg.reaction_s)
                traffic_cap = min(traffic_cap, stop)
                if stop < cap:
                    cap, state, reason = stop, 'YIELD', 'predicted crossing at s+%.1f m' % (
                        o['s'] - s_ego)

        path = []
        s = s_ego
        while s <= s_ego + cfg.path_length_m and s <= self.route.s[-1]:
            xy, _ = self.route.pose(s, offset_at(s))
            path.append(xy)
            s += cfg.path_step_m
        return Decision(state, path, cap, reason, offset_now, items, traffic_cap)
