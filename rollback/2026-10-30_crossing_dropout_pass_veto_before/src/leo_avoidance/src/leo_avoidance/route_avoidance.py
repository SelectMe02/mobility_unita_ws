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
from leo_avoidance.s4_path_search import search_offsets


@dataclass(frozen=True)
class RouteAvoidanceConfig:
    s4_wheelbase_m: float = 3.0
    s4_max_steering_rad: float = math.radians(29.)
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
    # Pedestrian hold (only inside pedestrian_hold_ranges): a person-sized
    # mover that made Ego yield keeps the stop until it is seen clear of the
    # band by this margin for release_scans LiDAR updates in a row.
    pedestrian_release_margin_m: float = 1.
    pedestrian_release_scans: float = 5.
    pedestrian_static_release_scans: float = 30.  # waiting at the kerb, outside the band
    pedestrian_lost_release_scans: float = 30.    # unseen after last being near the band
    pedestrian_adopt_m: float = 1.5     # a new track id this close continues the person
    pedestrian_approach_m: float = .05  # each lateral step toward the route confirming a crossing
    pedestrian_half_m: float = .6       # largest half extent held as a pedestrian
    pedestrian_min_half_m: float = .24  # a crossing person seen near is at least this big

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
    quintic: bool = False
    knots: tuple = ()
    values: tuple = ()
    slopes: tuple = ()

    def blend(self, t):
        t = max(0., min(1., t))
        return t*t*t*(10.-15.*t+6.*t*t) if self.quintic else smoothstep(t)

    def offset_at(self, s):
        if self.knots:
            if s <= self.knots[0]:
                return self.values[0]+self.slopes[0]*(s-self.knots[0])
            if s >= self.knots[-1]:
                return 0.
            i = min(len(self.knots)-2, max(0, int(np.searchsorted(self.knots, s))-1))
            h = self.knots[i+1]-self.knots[i]
            t = (s-self.knots[i])/h
            return ((2*t**3-3*t*t+1)*self.values[i] + (t**3-2*t*t+t)*h*self.slopes[i]
                    + (-2*t**3+3*t*t)*self.values[i+1] + (t**3-t*t)*h*self.slopes[i+1])
        if s <= self.ramp_start_s or s >= self.end_s:
            return 0.
        if s < self.full_s:
            return self.offset_m * self.blend(
                (s - self.ramp_start_s) / max(self.full_s - self.ramp_start_s, 1e-6))
        if s <= self.hold_end_s:
            return self.offset_m
        return self.offset_m * (1. - self.blend(
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
        self.pedestrian_hold_ranges = []
        self.pedestrian_hold = None
        self.pedestrian_holds = []
        self.observation_time = 0.
        self.person_d = {}  # track id -> route offsets at the previous two updates
        self.lane_changes = sorted(lane_changes, key=lambda v: v['start_s'])
        self.active = None
        self.last_s = None
        self.static_pass_zone = None
        self.pass_speed_25_ranges = []
        self.vehicle_follow_ranges = []  # S5 only: classify cars before static passing.
        self.r1_only_ranges = []
        self.static_memory = []
        self.static_pass_managed = False
        self.s4_motion_history = {}
        self.s4_mobile_until = {}
        self._s4_search_after = -math.inf

    def reset(self):
        self.active = None
        self.last_s = None
        self.crosswalk_hold = None
        self.pedestrian_hold = None
        self.pedestrian_holds = []
        self.observation_time = 0.
        self.person_d = {}
        self.static_memory = []
        self.static_pass_managed = False
        self.s4_motion_history = {}
        self.s4_mobile_until = {}
        self._s4_search_after = -math.inf

    def _s4_mobile_sources(self, items):
        """Remember measured motion before S4, without changing other planners.

        A velocity-classification dropout must not turn a moving car's trail
        into a permanent static footprint. Require common translation of both
        box edges, not expansion of one visible face, for geometric evidence.
        """
        now = self.observation_time
        self.s4_motion_history = {k: h for k, h in self.s4_motion_history.items()
                                  if now-h[-1][0] <= 1.}
        self.s4_mobile_until = {k: t for k, t in self.s4_mobile_until.items() if t >= now}
        for o in items:
            if o['id'] <= 0:
                continue
            history = self.s4_motion_history.setdefault(o['id'], [])
            if history and now <= history[-1][0]:
                continue
            if history and math.hypot(o['s']-history[-1][1], o['d']-history[-1][2]) > 5.:
                history.clear()
                self.s4_mobile_until.pop(o['id'], None)
            history.append((now, o['s'], o['d'], o['l'], o['r'], o['moving']))
            history[:] = [h for h in history if now-h[0] <= 1.]
            translated = False
            if len(history) >= 3 and now-history[0][0] >= .25:
                first, last = history[0], history[-1]
                for axis, extent in ((1, 3), (2, 4)):
                    low = (last[axis]-last[extent])-(first[axis]-first[extent])
                    high = (last[axis]+last[extent])-(first[axis]+first[extent])
                    common = min(abs(low), abs(high))
                    direction = math.copysign(1., last[axis]-first[axis])
                    consistent = all((b[axis]-a[axis])*direction >= -.1
                                     for a, b in zip(history, history[1:]))
                    if low*high > 0 and common >= .5 and common/(now-first[0]) >= .8 and consistent:
                        translated = True
            confirmed_motion = (len(history) >= 2 and o['moving'] and history[-2][5]
                                and now-history[-2][0] <= .5)
            if confirmed_motion or translated:
                self.s4_mobile_until[o['id']] = now+2.
        return set(self.s4_mobile_until)

    def set_static_pass_zone(self, start, full_start, full_end, end, offset):
        """Explicit saved S4 R2 corridor; approach/return stay within S4."""
        if not start < full_start < full_end < end or abs(offset) < 2.:
            raise ValueError('invalid S4 saved pass corridor')
        self.static_pass_zone = (start, full_start, full_end, end, offset)

    def _retain_static_blocks(self, items, s_ego):
        mobile_ids = self._s4_mobile_sources(items)
        zone = self.static_pass_zone
        self.static_pass_managed = bool(zone and zone[0] <= s_ego <= zone[3])
        if not self.static_pass_managed:
            self.static_memory = []
            return items
        # Stable multi-scan shapes survive occlusion until the rear clears.
        # A provisional single/jumping return only bridges a short dropout;
        # it must not become a permanent obstacle without confirmation.
        cfg = self.config
        self.static_memory = [o for o in self.static_memory
                              if s_ego-cfg.rear_extent_m <= o['s']+o['l']+cfg.post_gap_m
                              and (o.get('shape_confirmed', False)
                                   or self.observation_time-o.get('last_observed', self.observation_time) <= .6)]
        # Each memory has one source. A live object remains in `items` and
        # still undergoes all collision checks after its stale history clears.
        self.static_memory = [h for h in self.static_memory
                              if h['id'] not in mobile_ids]
        retained = []
        for h in self.static_memory:
            previous = h.get('evidence_stamp', -math.inf)
            fresh = self.observation_time > previous
            if self.observation_time-previous > .5:
                h['moving_seen'] = h['empty_seen'] = 0
                h['empty_since'] = None
            h['evidence_stamp'] = self.observation_time
            ids = h.get('source_ids', [h['id']])
            # A merged memory may contain another hidden static object.
            # Such memories need measured free space, not one moving ID.
            moving = len(ids) == 1 and any(o['moving'] and o['id'] > 0 and o['id'] in ids
                         and .5 < math.hypot(o['s']-h['s'], o['d']-h['d']) < 8.
                         for o in items)
            h['moving_seen'] = h.get('moving_seen', 0)+1 if fresh and moving else 0
            # A different, actually observed obstacle still occupying this
            # footprint prevents deletion merely because one source moved.
            occupied = any(not o['moving']
                           and abs(o['s']-h['s']) <= o['l']+h['l']+.2
                           and abs(o['d']-h['d']) <= o['r']+h['r']+.2 for o in items)
            observer = getattr(self, 'static_clearance_observer', None)
            clear = fresh and not occupied and observer is not None and observer(h)
            if clear:
                h['empty_seen'] = h.get('empty_seen', 0)+1
                if h.get('empty_since') is None:
                    h['empty_since'] = self.observation_time
            else:
                h['empty_seen'], h['empty_since'] = 0, None
            moved = h['moving_seen'] >= 2 and not occupied
            empty = (h['empty_seen'] >= 5 and h['empty_since'] is not None
                     and self.observation_time-h['empty_since'] >= .5)
            if not (moved or empty):
                retained.append(h)
        self.static_memory = retained
        for o in items:
            if (o['moving'] or o['id'] in mobile_ids or not (self._in_band(o, 0.) or self._in_band(o, zone[4])) or o['r'] < .4
                    or not zone[1] <= o['s'] <= zone[2]):
                continue
            match = next((h for h in self.static_memory
                          if h['id'] == o['id']
                          and abs(h['s']-o['s']) < 2. and abs(h['d']-o['d']) < 1.5), None)
            if match is None:
                self.static_memory.append(dict(o, v_s=0., v_d=0., remembered=True,
                                               source_ids=[o['id']], last_observed=self.observation_time,
                                               shape_confirmed=False))
            else:
                match['last_observed'] = self.observation_time
                ids = match.setdefault('source_ids', [match['id']])
                if o['id'] not in ids:
                    ids.append(o['id'])
                low, high = min(match['s']-match['l'], o['s']-o['l']), max(match['s']+match['l'], o['s']+o['l'])
                match['s'], match['l'] = (low+high)/2., (high-low)/2.
                match['r'] = max(match['r'], o['r']+abs(o['d']-match['d']))
        # Repeatedly measured geometry takes precedence over the same track's
        # accumulated envelope. Commit the same shape to memory, so loss on
        # the next scan cannot resurrect an obsolete inflated envelope.
        stable_shapes = []
        for o in items:
            # Use a fresh bounded observation window, not every view from
            # the last second. Keep the union of these views (not just the
            # smallest/latest face) when replacing the historical envelope.
            history = [h for h in self.s4_motion_history.get(o['id'], [])
                       if self.observation_time-h[0] <= .45]
            if (o['id'] > 0 and not o['moving'] and len(history) >= 3
                    and self.observation_time-history[0][0] >= .25
                    and all(not h[5] for h in history)
                    and all(b[0]-a[0] <= .2 for a, b in zip(history, history[1:]))
                    and all(np.ptp([h[axis] for h in history]) <= limit
                            for axis, limit in ((1, .35), (2, .35), (3, .25), (4, .25)))):
                if any(h['id'] == o['id'] for h in self.static_memory):
                    shape = dict(o, v_s=0., v_d=0., remembered=True,
                                 source_ids=[o['id']], last_observed=self.observation_time,
                                 shape_confirmed=True)
                    for axis, extent, center_key, size_key in ((1, 3, 's', 'l'), (2, 4, 'd', 'r')):
                        low = min(h[axis]-h[extent] for h in history)
                        high = max(h[axis]+h[extent] for h in history)
                        shape[center_key], shape[size_key] = (low+high)/2., (high-low)/2.
                    self.static_memory = [h for h in self.static_memory if h['id'] != o['id']]
                    self.static_memory.append(shape)
                    stable_shapes.append(shape)
        self._s4_reconcile_memory(items, stable_shapes)
        return items + [dict(h) for h in self.static_memory
                        if not any(o['id'] == h['id'] and self._s4_contains(o, h)
                                   for o in items if not o['moving'])]

    @staticmethod
    def _s4_contains(outer, inner):
        return all(abs(outer[c]-inner[c])+inner[e] <= outer[e]+1e-6
                   for c, e in (('s', 'l'), ('d', 'r')))

    @staticmethod
    def _s4_uncovered(old, observed):
        """Disjoint rectangles of old evidence outside a live footprint."""
        x0, x1 = old['s']-old['l'], old['s']+old['l']
        y0, y1 = old['d']-old['r'], old['d']+old['r']
        a, b = max(x0, observed['s']-observed['l']), min(x1, observed['s']+observed['l'])
        c, d = max(y0, observed['d']-observed['r']), min(y1, observed['d']+observed['r'])
        if a >= b or c >= d:
            return [old]
        result = []
        for lo, hi, bottom, top in ((x0, a, y0, y1), (b, x1, y0, y1),
                                    (a, b, y0, c), (a, b, d, y1)):
            if hi-lo > 1e-6 and top-bottom > 1e-6:
                result.append(dict(old, s=(lo+hi)/2., l=(hi-lo)/2.,
                                   d=(bottom+top)/2., r=(top-bottom)/2.))
        return result

    def _s4_reconcile_memory(self, items, stable_shapes):
        """Reconcile absent IDs using stable footprints and measured free space.

        Keep ambiguous/occluded evidence. Time or proximity alone never
        removes it. Replacement memory survives subsequent observation loss.
        """
        live_ids = {o['id'] for o in items}
        observer = getattr(self, 'static_clearance_observer', None)
        retained = []
        for h in self.static_memory:
            if h['id'] in live_ids:
                retained.append(h)
                continue
            matches = [o for o in stable_shapes if o['id'] != h['id']
                       and abs(o['s']-h['s']) < o['l']+h['l']
                       and abs(o['d']-h['d']) < o['r']+h['r']]
            accounted = False
            if len(matches) == 1:
                remaining = self._s4_uncovered(h, matches[0])
                accounted = not remaining or (observer is not None and all(
                    not any(abs(o['s']-part['s']) < o['l']+part['l']
                            and abs(o['d']-part['d']) < o['r']+part['r']
                            for o in items if not o['moving']) and observer(part)
                    for part in remaining))
            previous = h.get('reconcile_stamp', -math.inf)
            if self.observation_time > previous:
                if not accounted or self.observation_time-previous > .5:
                    h['reconcile_scans'], h['reconcile_since'] = 0, self.observation_time
                h['reconcile_stamp'] = self.observation_time
                if accounted:
                    h['reconcile_scans'] = h.get('reconcile_scans', 0)+1
            if (accounted and h.get('reconcile_scans', 0) >= 5
                    and self.observation_time-h.get('reconcile_since', self.observation_time) >= .5):
                continue
            retained.append(h)
        self.static_memory = retained

    def set_pedestrian_hold_ranges(self, ranges):
        """[(start_s, end_s)] on the base lap where crossing pedestrians are held."""
        ranges = [(float(a), float(b)) for a, b in ranges]
        if self.loop_length is not None:
            ranges += [(a + self.loop_length, b + self.loop_length) for a, b in ranges
                       if a < self.route.s[-1] - self.loop_length]
        self.pedestrian_hold_ranges = sorted(ranges)

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
            if self.pedestrian_hold is not None:
                self.pedestrian_hold['s'] -= self.loop_length
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

    def _person_sized(self, o):
        cfg = self.config
        return o['r'] <= cfg.pedestrian_half_m and o['l'] <= cfg.pedestrian_half_m

    def _held_person_sized(self, hold, o):
        # Admission remains strict. Only an already-associated track gets a
        # bounded 15 cm size hysteresis; position/motion matching still applies.
        if self._person_sized(o):
            return True
        limit = self.config.pedestrian_half_m + .15
        return (o['id'] in hold['ids'] and o['r'] <= limit and o['l'] <= limit
                and abs(o['l']-hold['l']) <= .15
                and abs(o['r']-hold.get('last_r', self.config.pedestrian_half_m)) <= .15)

    @staticmethod
    def _candidate_translation(a, b):
        """Lateral translation supported by both faces, not box growth.

        Samples are (time, s, d, half_length, half_width). Unknown motion
        is distinct from a measured stationary object.
        """
        dt = b[0]-a[0]
        if not .05 <= dt <= .5 or abs(b[1]-a[1]) > .5:
            return None
        delta = b[2]-a[2]
        growth = abs(b[4]-a[4])
        if (abs(b[3]-a[3]) > .15 or growth > .15
                or growth > max(.025, .5*abs(delta))):
            return None
        vd = delta/dt
        return vd if abs(vd) <= 4.5 else None

    def _candidate_velocity(self, hold):
        vd = hold.get('vd', 0.)
        age = self.observation_time-hold.get('velocity_at', hold.get('seen_at', self.observation_time))
        if age <= .5:
            return vd
        # Retain a possible occluded crossing at the body; never extrapolate
        # an old roadside candidate across the entire road indefinitely.
        band = self.config.band(hold.get('last_r', self.config.pedestrian_half_m))+.3
        projected = hold['d']+vd*.5
        if hold['d']*projected > 0. and min(abs(hold['d']), abs(projected)) > band:
            return 0.
        return vd

    def _hold_match(self, hold, o):
        dt = max(0., self.observation_time-hold.get('seen_at', self.observation_time))
        if abs(o['s']-hold['s']) > 1.5:
            return False
        if hold.get('tentative') and not hold.get('confirmed_crossing'):
            # Same-ID observations must be able to correct stale velocity.
            # A new ID needs local, recent evidence, not a long extrapolation.
            if o['id'] in hold['ids']:
                return abs(o['d']-hold['d']) <= 1.5
            if dt > .5 or abs(o['d']-hold['d']) > .75:
                return False
            if o['d']*hold['d'] < 0. and min(abs(o['d']), abs(hold['d'])) > .5:
                return False
        if dt > 3.01:
            # A stale velocity must not extrapolate indefinitely. Reconnect
            # only a measured person close to the last occupied footprint;
            # clear-space observations handle a person that has left it.
            return abs(o['d']-hold['d']) <= .75
        vd = hold.get('vd', 0.)
        expected = hold['d'] + vd*dt
        if abs(o['d']-expected) > 1.5 + .3*dt:
            return False
        if o['id'] not in hold['ids']:
            if abs(vd) > .2 and (o['d']-hold['d'])*vd < -.2:
                return False
            if abs(o['v_d']) > .2 and o['v_d']*vd < -.2:
                return False
        return True

    def _early_pedestrian_approach(self, o, speed, front_s):
        """Small, consistently approaching object before velocity confirmation.

        Only a predicted overlap near the stopping horizon earns an early
        candidate; stationary, departing and distant objects do not.
        """
        cfg = self.config
        h = self.s4_motion_history.get(o['id'], [])[-3:]
        if len(h) < 3 or self.observation_time-h[-1][0] > .2:
            return None
        rates = []
        for a, b in zip(h, h[1:]):
            dt = b[0]-a[0]
            if (not .05 <= dt <= .25 or a[2]*b[2] <= 0.
                    or abs(a[2])-abs(b[2]) < .05
                    or abs(a[1]-b[1]) > .5
                    or abs(a[3]-b[3]) > .15 or abs(a[4]-b[4]) > .15):
                return None
            rate = self._candidate_translation(a, b)
            if rate is None:
                return None
            rates.append(rate)
        vd = sum(rates)/len(rates)
        if not .4 <= abs(vd) <= 4.5 or abs(rates[0]-rates[1]) > 1.5:
            return None
        v = max(speed, 3.)  # evaluate a modest start speed while waiting at a signal
        ahead = o['s']-o['l']-front_s
        band = cfg.band(o['r'])
        entry = max(0., (abs(o['d'])-band)/abs(vd))
        leave = (abs(o['d'])+band)/abs(vd)
        arrival = max(0., ahead)/v
        horizon = v*cfg.reaction_s+v*v/(2.*cfg.brake_mps2)+v+cfg.stop_gap_m
        if (0. < ahead <= horizon and entry <= min(3., arrival+cfg.arrival_after_s)
                and leave >= max(0., arrival-cfg.arrival_before_s)):
            return vd
        return None

    def _note_pedestrian(self, o, binding=True, tentative=False, approach_vd=None):
        """Start or refresh the pedestrian hold for a mover that made Ego yield.

        Only a person-sized candidate whose position really moved toward the
        route over the last two updates, in the direction of its estimated
        velocity, starts a hold. Roadside clutter with a stale velocity or a
        single jump does not. A predicted crossing must also demand braking
        now (binding). Tentative S5 candidates use measured approach without
        requiring the velocity classifier, and retain the stop through loss.
        """
        cfg = self.config
        for existing in self.pedestrian_holds:
            if self._hold_match(existing, o):
                return
        previous_hold = self.pedestrian_hold
        self.pedestrian_hold = None
        if (not binding or not self._person_sized(o)
                or max(o['r'], o['l']) < cfg.pedestrian_min_half_m
                or not any(a <= o['s'] <= b for a, b in self.pedestrian_hold_ranges)
                or (abs(o['v_s']) >= cfg.moving_speed_mps
                    and abs(o['v_s']) >= 2. * abs(o['v_d']))):
            self.pedestrian_hold = previous_hold
            return  # traffic along the lane is not a crossing pedestrian
        if tentative and abs(o['d']) > cfg.band(o['r'])+.3:
            samples = self.s4_motion_history.get(o['id'], [])[-2:]
            if len(samples) == 2:
                a, b = samples
                vd = self._candidate_translation(a, b)
                if vd is None or o['d']*vd >= 0. or abs(vd) < .4:
                    self.pedestrian_hold = previous_hold
                    return
        history = self.person_d.get(o['id'], ())
        if self.pedestrian_hold is None:
            path = list(history) + [o['d']]
            if tentative:
                path = path[-2:]
            if approach_vd is None and (len(path) < (2 if tentative else 3) or any(
                    abs(a) - abs(b) < (max(.25, cfg.pedestrian_approach_m)
                                       if tentative else cfg.pedestrian_approach_m)
                    or (not tentative and o['v_d'] * (b - a) <= 0.)
                    for a, b in zip(path, path[1:]))):
                self.pedestrian_hold = previous_hold
                return
        if self.pedestrian_hold is None:
            self.pedestrian_hold = dict(ids=set(), s=o['s'], d=o['d'], l=o['l'],
                                        cap=math.inf, full_brake=False,
                                        clear_scans=0, static_scans=0,
                                        lost_scans=0, near=True,
                                        confirmed_crossing=not tentative,
                                        entry_side=math.copysign(1., o['d']))
            if tentative:
                self.pedestrian_hold.update(tentative=True, entry_side=math.copysign(1., o['d']),
                                            approach_scans=1, stationary_scans=0,
                                            initial_d=o['d'], last_r=o['r'], last_l=o['l'])
        if tentative and approach_vd is None:
            measured = self.s4_motion_history.get(o['id'], [])[-2:]
            if len(measured) == 2:
                a, b = measured
                dt = b[0]-a[0]
                if .05 <= dt <= .25:
                    vd = self._candidate_translation(a, b)
                    if vd is not None and .4 <= abs(vd) <= 4.5 and o['d']*vd < 0.:
                        approach_vd = vd
        self.pedestrian_hold.update(seen_at=self.observation_time, velocity_at=self.observation_time,
                                    vd=o['v_d'] if approach_vd is None else approach_vd)
        self.pedestrian_hold['ids'].add(o['id'])
        self.pedestrian_holds.append(self.pedestrian_hold)

    def _pedestrian_hold_cap(self, items, speed, s_ego, front_s, offset_at):
        remaining = []
        used = set()
        for hold in self.pedestrian_holds:
            self.pedestrian_hold = hold
            cap = self._single_pedestrian_hold_cap(
                [o for o in items if o['id'] not in used], speed, s_ego, front_s, offset_at)
            if hold.get('seen_at') == self.observation_time and 'observed_id' in hold:
                used.add(hold['observed_id'])
            if cap is not None:
                remaining.append((cap, hold))
        self.pedestrian_holds = [h for _, h in remaining]
        self.pedestrian_hold = min(remaining, key=lambda entry: entry[0])[1] if remaining else None
        return min((cap for cap, _ in remaining), default=None)

    def _single_pedestrian_hold_cap(self, items, speed, s_ego, front_s, offset_at):
        """Speed cap while a crossing pedestrian is held, or None.

        A momentary velocity dropout or track loss does not release it; only
        seeing the person clear of the band (or Ego past it) does. While held
        the cap never rises, and a stop that the planning deceleration cannot
        make is a full stop.
        """
        hold = self.pedestrian_hold
        if hold is None:
            return None
        cfg = self.config
        near = [o for o in items if self._hold_match(hold, o) and (
            self._held_person_sized(hold, o)
            or (hold.get('tentative') and not hold.get('confirmed_crossing')
                and o['id'] in hold['ids']
                and abs(o['d']-offset_at(o['s'])) > cfg.band(o['r'])+.3))]
        # A growing same-ID roadside box is still an observation. It must
        # invalidate old candidate motion instead of looking like track loss.
        if near:
            o = min(near, key=lambda o: math.hypot(o['s'] - hold['s'], o['d'] - hold['d']))
            previous_d, previous_s = hold['d'], hold['s']
            dt = self.observation_time-hold.get('seen_at', self.observation_time)
            fresh = 0. < dt <= .5
            measured_vd = ((o['d']-previous_d)/dt if dt > .05 else hold.get('vd', 0.))
            if hold.get('tentative') and not hold.get('confirmed_crossing'):
                if fresh:
                    rate = self._candidate_translation(
                        (self.observation_time-dt, previous_s, previous_d,
                         hold.get('last_l', hold['l']), hold.get('last_r', o['r'])),
                        (self.observation_time, o['s'], o['d'], o['l'], o['r']))
                    hold['vd'] = 0. if rate is None else rate
                    hold['velocity_at'] = self.observation_time
            elif abs(o['v_d']) > .2:
                hold['vd'] = o['v_d']
            elif not hold.get('tentative') and .2 < abs(measured_vd) <= 4.5:
                hold['vd'] = measured_vd
            hold['seen_at'] = self.observation_time
            hold['observed_id'] = o['id']
            hold['ids'].add(o['id'])
            hold.update(s=o['s'], d=o['d'], l=o['l'], lost_scans=0)
            lateral = abs(o['d'] - offset_at(o['s']))
            hold['near'] = lateral < cfg.band(o['r']) + cfg.pedestrian_release_margin_m
            in_band = lateral < cfg.band(o['r'])
            hold['static_scans'] = (0 if o['moving'] or in_band
                                    else hold['static_scans'] + 1)
            occupied = hold['near']
            if hold.get('tentative'):
                delta = abs(previous_d)-abs(o['d'])
                consistent = (self._candidate_translation(
                    (self.observation_time-dt, previous_s, previous_d, hold['last_l'], hold['last_r']),
                    (self.observation_time, o['s'], o['d'], o['l'], o['r'])) is not None
                              and abs(o['r']-hold['last_r']) < .15
                              and abs(o['l']-hold['last_l']) < .15
                              and abs(o['s']-previous_s) < .5)
                hold['approach_scans'] = (hold['approach_scans']+1
                    if consistent and delta >= .1 else 0)
                hold['stationary_scans'] = (hold['stationary_scans']+1
                    if fresh and consistent and lateral > cfg.band(o['r']) + .3 and not o['moving']
                    and abs(o['d']-previous_d) < .1 else 0)
                relative_d = o['d']-offset_at(o['s'])
                previous_relative_d = previous_d-offset_at(previous_s)
                # An unconfirmed candidate can leave without ever crossing.
                # Require measured outward displacement, not a velocity flag.
                outward = (fresh and consistent and lateral > cfg.band(o['r'])+.3
                           and relative_d*previous_relative_d > 0.
                           and lateral-abs(previous_relative_d) >= .05)
                hold['departing_scans'] = hold.get('departing_scans', 0)+1 if outward else 0
                hold.update(last_r=o['r'], last_l=o['l'])
                # Candidate slowdown is reversible. Confirmed path occupancy
                # or sustained approach upgrades it to a persistent hazard.
                confirmed = (in_band or (o['moving'] and o['d']*o['v_d'] < 0.)
                             or (hold['approach_scans'] >= 2
                                 and abs(hold['initial_d'])-abs(o['d']) >= .4))
                if confirmed:
                    hold.update(tentative=False, confirmed_crossing=True)
                elif hold['stationary_scans'] >= 4 or hold['departing_scans'] >= 3:
                    self.pedestrian_hold = None
                    return None
                occupied = True
                hold['static_scans'] = 0
            if hold.get('confirmed_crossing'):
                relative_d = o['d']-offset_at(o['s'])
                outside = abs(relative_d) > cfg.band(o['r']) + .2
                outward = relative_d*hold.get('vd', 0.) > .1
                occupied = not (outside and outward)
        else:
            # Unseen (e.g. right at the bumper): still where it was last seen.
            hold['lost_scans'] += 1
            if hold.get('tentative') and not hold.get('confirmed_crossing'):
                # A never-confirmed off-path candidate is not a hidden
                # crossing pedestrian. Keep a short dropout bridge only
                # when its last measured approach could still reach us.
                relative = hold['d']-offset_at(hold['s'])
                band = cfg.band(hold.get('last_r', cfg.pedestrian_half_m))
                vd = self._candidate_velocity(hold)
                approaching = relative*vd < -.1
                soon = approaching and abs(relative)-abs(vd)*.8 <= band+.3
                if (self.observation_time-hold['seen_at'] >= .6
                        and abs(relative) > band+.3 and not soon):
                    self.pedestrian_hold = None
                    return None
            occupied = (hold['near']
                        and hold['lost_scans'] < cfg.pedestrian_lost_release_scans)
            if hold.get('tentative') or hold.get('confirmed_crossing'):
                occupied = True
            observer = getattr(self, 'clearance_observer', None)
            previous_clear_stamp = hold.get('clear_observation_stamp', -math.inf)
            fresh = self.observation_time > previous_clear_stamp
            if self.observation_time - previous_clear_stamp > .5:
                hold['visible_clear_since'] = None
                hold['visible_clear_scans'] = 0
            hold['clear_observation_stamp'] = self.observation_time
            # A clear old footprint alone cannot release an approaching
            # second pedestrian, a stationary occupant, or a new track ID.
            current_risk = False
            for current in items:
                if (current['s']+current['l'] < s_ego-cfg.rear_extent_m
                        or current['s']-current['l'] > hold['s']+cfg.stop_gap_m):
                    continue
                lateral = current['d']-offset_at(current['s'])
                band_now = cfg.band(current['r'])
                velocity = current.get('v_d', 0.)
                approaching_now = (lateral*velocity < 0. and
                    abs(lateral)-abs(velocity)*cfg.predict_s <= band_now)
                if abs(lateral) <= band_now or approaching_now:
                    current_risk = True
                    break
            visible_clear = (not current_risk and fresh and speed <= .3 and observer is not None
                             and self.observation_time-hold.get('seen_at', self.observation_time) >= .6
                             and observer(hold, offset_at))
            if visible_clear:
                if hold.get('visible_clear_since') is None:
                    hold['visible_clear_since'] = self.observation_time
                hold['visible_clear_scans'] = hold.get('visible_clear_scans', 0) + 1
                if (hold['visible_clear_scans'] >= 5
                        and self.observation_time-hold['visible_clear_since'] >= .5):
                    self.pedestrian_hold = None
                    return None
            else:
                hold['visible_clear_since'] = None
                hold['visible_clear_scans'] = 0
        if near:
            hold['visible_clear_since'] = None
            hold['visible_clear_scans'] = 0
        hold['clear_scans'] = 0 if occupied else hold['clear_scans'] + 1
        if (hold['s'] + hold['l'] < s_ego
                or hold['clear_scans'] >= (3 if hold.get('confirmed_crossing') else cfg.pedestrian_release_scans)
                or hold['static_scans'] >= cfg.pedestrian_static_release_scans):
            self.pedestrian_hold = None
            return None
        ahead = hold['s'] - hold['l'] - front_s
        if hold.get('tentative'):
            relative = hold['d']-offset_at(hold['s'])
            band = cfg.band(hold.get('last_r', cfg.pedestrian_half_m))
            if abs(relative) > band+.3:
                vd = self._candidate_velocity(hold)
                approaching = relative*vd < 0. and abs(vd) >= .2
                arrival = max(0., ahead)/max(speed, 3.)
                entry = max(0., (abs(relative)-band)/max(abs(vd), .01))
                leave = (abs(relative)+band)/max(abs(vd), .01)
                overlap = (approaching and entry <= min(cfg.predict_s, arrival+cfg.arrival_after_s)
                           and leave >= max(0., arrival-cfg.arrival_before_s))
                if not overlap:
                    # Preserve observation history without braking for a
                    # longitudinal distance to something outside our path.
                    hold['cap'] = math.inf
                    return math.inf
            # A candidate outside the path limits approach speed, without
            # permanently latching a zero cap from one noisy displacement.
            hold['cap'] = min(10./3.6, stopping_speed(max(0., ahead-cfg.stop_gap_m),
                                                    cfg.brake_mps2, cfg.reaction_s))
            return hold['cap']
        stop = (0. if ahead <= 0. else
                stopping_speed(ahead - cfg.stop_gap_m + 2., cfg.brake_mps2, cfg.reaction_s))
        if hold['full_brake'] or stop + .3 < speed:
            hold['full_brake'] = speed > .3      # keep full braking until stopped
            stop = 0.
        hold['cap'] = min(hold['cap'], stop)
        return hold['cap']

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

    def s4_curvature_limit(self):
        return math.tan(self.config.s4_max_steering_rad)/self.config.s4_wheelbase_m

    def _s4_swept_check(self, profile, items, s_ego, speed, check_curvature=True):
        """Check oriented ego body against obstacle rectangles on the route.

        Sample the complete entry, hold and return, including rear overhang.
        A small sampling allowance is added to the existing safety margin.
        """
        cfg = self.config
        peak = 0.
        for s in np.arange(s_ego, profile.end_s + .26, .25):
            d = profile.offset_at(s)
            left, right = profile.offset_at(s-.05), profile.offset_at(s+.05)
            slope = (right-left)/.1
            curvature = abs((right-2*d+left)/.0025)/(1+slope*slope)**1.5
            peak = max(peak, curvature)
            if check_curvature and peak > (self.s4_curvature_limit() if self.static_pass_managed else .14):
                return False, None, s, peak
            angle = math.atan(slope)
            c, sn = math.cos(angle), math.sin(angle)
            half_l = (cfg.front_extent_m+cfg.rear_extent_m)/2.+.15
            half_w = cfg.vehicle_half_width_m+(.15 if self.static_pass_managed else cfg.side_margin_m+.15)
            center = (cfg.front_extent_m-cfg.rear_extent_m)/2.
            es, ed = s+c*center, d+sn*center
            t = max(0., s-s_ego)/max(speed, 1.)
            for o in items:
                ds, dd = o['s']+o['v_s']*t-es, o['d']+o['v_d']*t-ed
                ol, ow = o['l'], o['r']
                if (abs(ds) <= half_l*abs(c)+half_w*abs(sn)+ol
                        and abs(dd) <= half_l*abs(sn)+half_w*abs(c)+ow
                        and abs(ds*c+dd*sn) <= half_l+ol*abs(c)+ow*abs(sn)
                        and abs(-ds*sn+dd*c) <= half_w+ol*abs(sn)+ow*abs(c)):
                    return False, o, s, peak
        return True, None, profile.end_s, peak

    def _s4_candidate(self, items, s_ego, speed, block_first, last_end):
        start, full_start, full_end, end, offset = self.static_pass_zone
        hold = last_end+self.config.rear_extent_m+self.config.post_gap_m
        # Full R2 is preferred; partial offsets stay inside the same saved
        # corridor and can avoid a harmless object farther into that lane.
        for scale in (1., .85, .65, .5):
            for full in np.linspace(max(full_start, block_first-self.config.front_extent_m-2.),
                                    min(full_end, last_end), 5):
                for length in (14., 11., 8.):
                    ramp_start = max(start, s_ego, full-length)
                    hold_end = max(hold, full)
                    end_s = min(end, hold_end+14.)
                    if full-ramp_start < 2. or end_s-hold_end < 2. or hold_end > full_end:
                        continue
                    candidate = _Pass(ramp_start, full, hold_end, end_s,
                                      offset*scale, block_first, True)
                    # First bound curvature without spending time on objects.
                    if not self._s4_swept_check(candidate, [], s_ego, speed)[0]:
                        continue
                    # Check both current speed and the intended low pass speed.
                    safe = all(self._s4_swept_check(candidate, items, s_ego, v)[0]
                               for v in (speed, min(speed, 3.)))
                    if safe:
                        # Prefer a slightly later entry without compressing
                        # the curve: preserve its length and the return anchor.
                        for delay in (2., 1.):
                            shifted = replace(candidate, ramp_start_s=ramp_start+delay,
                                              full_s=full+delay)
                            if (shifted.full_s <= min(hold_end, full_end)
                                    and all(self._s4_swept_check(shifted, items, s_ego, v)[0]
                                            for v in (speed, min(speed, 3.)))):
                                return shifted
                        # A small S4-only timing adjustment: retain the proven
                        # full-offset/return anchors and delay entry by 1 m.
                        # If the shorter entry cannot pass the same checks,
                        # keep the original safe candidate.
                        delayed = replace(candidate, ramp_start_s=ramp_start+1.)
                        if (full-delayed.ramp_start_s >= 2.
                                and all(self._s4_swept_check(delayed, items, s_ego, v)[0]
                                        for v in (speed, min(speed, 3.)))):
                            return delayed
                        return candidate
        return self._s4_adaptive_candidate(items, s_ego, speed, block_first)

    def _s4_adaptive_candidate(self, items, s_ego, speed, block_first):
        # Only static S4 geometry is optimized. Existing traffic checks still
        # handle live movers; never plan a spline through their future path.
        if any(o['moving'] for o in items):
            return None
        relevant = [o for o in items if o['r'] >= .4 and o['s'] > s_ego
                    and (self._in_band(o, 0.) or self._in_band(o, self.static_pass_zone[4]))]
        continuing = self.active is not None and bool(self.active.knots)
        if not continuing and not any(a['id'] != b['id'] and abs(a['d']-b['d']) > 1.6
                   and abs(a['s']-b['s']) > 3. for a in relevant for b in relevant):
            return None  # Preserve the established single-block S4 behaviour.
        if self.observation_time < getattr(self, '_s4_search_after', -math.inf):
            return None
        self._s4_search_after = self.observation_time+.5
        initial = self.active.offset_at(s_ego) if self.active else 0.
        slope = ((self.active.offset_at(s_ego+.05)-self.active.offset_at(s_ego-.05))/.1
                 if self.active else 0.)
        end, lane = self.static_pass_zone[3:]
        solution = search_offsets(s_ego, end, initial, slope, lane, items, self.config)
        if solution is None:
            return None
        knots, values, slopes = solution
        path = _Pass(s_ego, s_ego+1., end-1., end, lane, block_first,
                     knots=tuple(knots), values=tuple(values), slopes=tuple(slopes))
        if self._s4_swept_check(path, items, s_ego, speed)[0]:
            return path
        return None

    def _s4_curve_speed(self, profile, s_ego, speed):
        """Curvature-dependent speed with braking room for upcoming bends."""
        cap = math.inf
        for s in np.arange(s_ego, profile.end_s+.01, .5):
            d = profile.offset_at(s)
            left, right = profile.offset_at(s-.05), profile.offset_at(s+.05)
            slope = (right-left)/.1
            curvature = abs(right-2*d+left)/.0025/(1+slope*slope)**1.5
            bend_speed = math.sqrt(3.1/max(curvature, .001))
            room = max(0., s-s_ego-speed*self.config.reaction_s)
            cap = min(cap, math.sqrt(bend_speed**2+2*self.config.brake_mps2*room))
        return cap

    def update(self, ego_xy, ego_speed_mps, obstacles, stop_line_distance_m=None,
               stop_line_permitted=True, stamp=None, clearance_observer=None,
               static_clearance_observer=None):
        """Return the offset path (map points), a speed cap and the state."""
        self.observation_time = (float(stamp) if stamp is not None else self.observation_time + .1)
        self.clearance_observer = clearance_observer
        self.static_clearance_observer = static_clearance_observer
        cfg = self.config
        speed = max(0., ego_speed_mps if math.isfinite(ego_speed_mps) else 0.)
        s_ego, _ = self._locate_ego(*ego_xy)
        if any(a <= s_ego < b for a, b in self.pass_speed_25_ranges):
            cfg = replace(cfg, pass_speed_kmh=25.)
        items = self._to_route(obstacles, s_ego)
        items = self._retain_static_blocks(items, s_ego)
        r1_only = any(a <= s_ego <= b for a, b in self.r1_only_ranges)
        if r1_only:
            self.active = None
        front_s = s_ego + cfg.front_extent_m
        state, reason = 'FOLLOW', 'clear'
        cap = math.inf
        traffic_cap = math.inf

        if self.active is not None and s_ego > self.active.end_s:
            self.active = None
        offset_now = self.active.offset_at(s_ego) if self.active else 0.

        # Static blocks in Ego's own lane ahead of the bumper (or beside it while passing).
        follow_zone = any(a <= s_ego < b for a, b in self.vehicle_follow_ranges)
        def static_pass_candidate(o):
            # A brief static label must not turn a recent crosser into a pass.
            if (not self.static_pass_managed
                    and any(a <= o['s'] <= b for a, b in self.pedestrian_hold_ranges)
                    and o['id'] in self.s4_mobile_until):
                return False
            if not follow_zone or o['l'] < 1.25:
                return True
            # Initial/dropout static labels must not start a car overtake.
            # This only gates path creation: all obstacles remain in braking
            # and pedestrian/crossing checks below, with their original data.
            history = self.s4_motion_history.get(o['id'], [])
            return (o['id'] not in self.s4_mobile_until
                    and abs(o['v_s']) < cfg.moving_speed_mps
                    and len(history) >= 3 and history[-1][0]-history[0][0] >= .6
                    and max(h[1] for h in history)-min(h[1] for h in history) < .3
                    and not any(h[5] for h in history))
        blocks = sorted((o for o in items if not o['moving'] and static_pass_candidate(o) and self._in_band(o, 0.)
                         and o['s'] + o['l'] > front_s - (cfg.front_extent_m
                                                           if self.active else 0.)),
                        key=lambda o: o['s'])
        if self.active is not None:
            if self.static_pass_managed and not self._s4_swept_check(self.active, items, s_ego, speed)[0]:
                replacement = self._s4_adaptive_candidate(items, s_ego, speed, self.active.block_first_s)
                if replacement is not None:
                    self.active = replacement
                    offset_now = self.active.offset_at(s_ego)
            # A further block soon after the current one extends the hold.
            for o in blocks:
                if self.active.knots:
                    break  # The complete multi-bend path has already been checked.
                if o['s'] - o['l'] <= self.active.hold_end_s + cfg.merge_gap_m:
                    needed_hold = o['s'] + o['l'] + cfg.rear_extent_m + cfg.post_gap_m
                    if self.static_pass_managed and needed_hold > self.active.hold_end_s:
                        minimum_return = math.sqrt(5.8*abs(self.active.offset_m)/self.s4_curvature_limit())
                        if (needed_hold > self.static_pass_zone[2]
                                or needed_hold+minimum_return > self.static_pass_zone[3]):
                            cap = 0.  # No safe return inside the saved S4 corridor.
                            continue
                    self.active.hold_end_s = max(
                        self.active.hold_end_s,
                        needed_hold)
                    self.active.end_s = max(self.active.end_s,
                                            self.active.hold_end_s + cfg.ramp_out_m)
                    if self.static_pass_managed:
                        self.active.end_s = min(self.active.end_s, self.static_pass_zone[3])
            clear, blocker = self._lane_clear(items, self.active.offset_m, s_ego,
                                              self.active.end_s, speed)
            if self.static_pass_managed:
                clear, blocker, hit_s, _ = self._s4_swept_check(self.active, items, s_ego, speed)
                if not clear:
                    cap = min(cap, stopping_speed(hit_s-s_ego, cfg.brake_mps2, cfg.reaction_s))
            elif any(a <= s_ego <= b for a, b in self.pedestrian_hold_ranges):
                # In S5 a stationary kerb-side person must intersect the
                # actual remaining entry/return, not the old full-offset
                # lane strip. Keep moving traffic's arrival checks intact.
                clear, blocker = self._lane_clear(
                    [o for o in items if o['moving']], self.active.offset_m,
                    s_ego, self.active.end_s, speed)
                static_clear, static_blocker, _, _ = self._s4_swept_check(
                    self.active, [o for o in items if not o['moving']],
                    s_ego, speed, check_curvature=False)
                if not static_clear:
                    clear, blocker = False, static_blocker
            beside = front_s >= self.active.block_first_s - cfg.pre_gap_m
            protected_pass = self.static_pass_managed or any(a <= s_ego <= b for a, b in self.pedestrian_hold_ranges)
            if not clear and not beside and not protected_pass:
                self.active = None
                offset_now = 0.
                reason = 'pass lane blocked at s+%.1f m; return and wait' % (
                    blocker['s'] - s_ego)
            elif not clear:
                gap = (blocker['s'] - blocker['l'] - front_s - cfg.stop_gap_m
                       if blocker is not None else 0.)
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
            if r1_only:
                offsets = []
            if self.static_pass_managed:
                offsets = []  # S4 searches swept paths below, not lane strips.
            clear, blocker, pass_offset = False, None, None
            for offset in offsets:
                clear, blocker = self._lane_clear(items, offset, s_ego, end_s, speed)
                if clear:
                    pass_offset = offset
                    break
            s4_candidate = None
            if self.static_pass_managed:
                s4_candidate = self._s4_candidate(items, s_ego, speed, block_first_s, last_end)
                clear = s4_candidate is not None
                if clear:
                    ramp_start, full = s4_candidate.ramp_start_s, s4_candidate.full_s
                    hold_end, end_s = s4_candidate.hold_end_s, s4_candidate.end_s
                    pass_offset, creep = s4_candidate.offset_m, False
            stop_line_in_pass = (stop_line_distance_m is not None
                                 and not stop_line_permitted
                                 and 0. <= stop_line_distance_m
                                 <= end_s - front_s + cfg.stop_line_clear_m)
            gap = block_first_s - front_s - cfg.stop_gap_m
            side = lambda offset: 'left' if offset > 0 else 'right'
            if clear and not stop_line_in_pass and (not creep or speed < .3):
                self.active = s4_candidate or _Pass(ramp_start, full, hold_end, end_s,
                                                   pass_offset, block_first_s, self.static_pass_managed)
                offset_now = self.active.offset_at(s_ego)
                state = 'PASSING'
                reason = ('pulling out to the %s lane from behind a static block' if creep
                          else 'passing static block on the %s lane') % side(pass_offset)
                if creep:
                    # Waiting close behind it: creep out over the room that is left.
                    cap = min(cap, 7. / 3.6)
            else:
                cap = min(cap, stopping_speed(gap, cfg.brake_mps2, cfg.reaction_s))
                if self.static_pass_managed:
                    # Preserve entry room while waiting, instead of consuming
                    # it until only the bumper stopping gap remains.
                    needed_offset = max(.5, cfg.band(first['r'])-abs(first['d']))
                    entry = full_s-math.sqrt(5.8*needed_offset/self.s4_curvature_limit())
                    cap = min(cap, stopping_speed(entry-s_ego-1., cfg.brake_mps2, cfg.reaction_s))
                if stop_line_in_pass:
                    traffic_cap = min(traffic_cap, cap)
                if not offsets:
                    state, reason = ('WAIT_PASS', 'S4 saved corridor has insufficient pass room') if self.static_pass_managed else ('WAIT_SOLID_LINE', 'solid line beside the block; waiting')
                else:
                    state = 'WAIT_PASS'
                    reason = ('red stop line inside the pass' if stop_line_in_pass else
                              '%s lane occupied at s+%.1f m' % (side(offsets[-1]), blocker['s'] - s_ego)
                              if blocker is not None else 'too close to pull out; slowing')
                if self.static_pass_managed and not stop_line_in_pass:
                    state, reason = 'WAIT_PASS', 'S4 no collision-free swept path; preserving entry room'
        if self.active is not None:
            cap = min(cap, cfg.pass_speed_kmh / 3.6)
            if self.active.knots:
                cap = min(cap, self._s4_curve_speed(self.active, s_ego, speed))
            if self.active.quintic:
                length = min(self.active.full_s-self.active.ramp_start_s,
                             self.active.end_s-self.active.hold_end_s)
                curvature_bound = 5.8*abs(self.active.offset_m)/max(length*length, .01)
                cap = min(cap, 25./3.6, math.sqrt(3.1/max(curvature_bound, .001)))

        def offset_at(s):
            return self.active.offset_at(s) if self.active else 0.

        # Anything static left on the planned path must still be stoppable for.
        for o in items:
            if (self.static_pass_managed and self.active is not None
                    and o['s']-o['l'] <= self.active.end_s+cfg.front_extent_m):
                continue  # Complete oriented body check above owns this path.
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
                if (self.static_pass_managed and self.active is not None
                        and not o['moving']
                        and self.static_pass_zone[0] <= o['s'] - o['l']
                        and o['s'] + o['l'] <= min(self.static_pass_zone[3],
                                                  self.active.end_s)):
                    # S4 contains static fixtures: the oriented swept-path
                    # check above owns their clearance and braking. Do not
                    # reinterpret them as pedestrians by size alone on the
                    # global centerline. Outside S4 and movers stay protected.
                    continue
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

        # In the configured S5 pedestrian zone, measured lateral approach
        # can justify braking before the track's dynamic label is confirmed.
        if self.active is None:
            for o in items:
                if (0. < o['s'] - front_s < 35. and abs(o['d']) < 5.
                        and not o['moving']):
                    self._note_pedestrian(o, tentative=True,
                        approach_vd=self._early_pedestrian_approach(o, speed, front_s))

        # Geometry-based early braking also covers fragmented/wide detections
        # that cannot enter the strict person-sized hold. No persistent latch:
        # only fresh, coherent approach and overlapping arrival times qualify.
        if self.active is None and not self.static_pass_managed:
            for o in items:
                if (any(a <= o['s'] <= b for a, b in self.pedestrian_hold_ranges)
                        and o['l'] <= .6 and o['r'] <= 1.2
                        and self._early_pedestrian_approach(o, speed, front_s) is not None):
                    stop = stopping_speed(o['s']-o['l']-front_s-cfg.stop_gap_m,
                                          cfg.brake_mps2, cfg.reaction_s)
                    traffic_cap = min(traffic_cap, stop)
                    if stop < cap:
                        cap, state, reason = stop, 'YIELD', 'measured crossing approach'

        # Movers: lead traffic in the path and predicted crossings.
        for o in items:
            if not o['moving']:
                continue
            crossing_zone = (not self.static_pass_managed
                and any(a <= o['s'] <= b for a, b in self.pedestrian_hold_ranges)
                and abs(o['v_d']) >= cfg.moving_speed_mps
                and abs(o['v_d']) > abs(o['v_s']))
            crossing_extra = 0. if crossing_zone else 2.
            off = offset_at(o['s'])
            ahead = o['s'] - o['l'] - front_s
            if ahead < -cfg.front_extent_m:
                continue
            if self._in_band(o, off):
                if ahead <= 0.:
                    traffic_cap = 0.
                    cap, state, reason = 0., 'YIELD', 'mover beside the bumper'
                    self._note_pedestrian(o)
                    continue
                if o['v_s'] > cfg.moving_speed_mps and abs(o['v_d']) < cfg.moving_speed_mps:
                    # Lead traffic: keep a gap that still stops if it brakes hard.
                    lead = stopping_speed(ahead - cfg.follow_gap_m, cfg.brake_mps2,
                                          cfg.reaction_s) + o['v_s']
                    traffic_cap = min(traffic_cap, lead)
                    if lead < cap:
                        cap, state, reason = lead, 'FOLLOW_LEAD', 'following a lead mover'
                    continue
                stop = stopping_speed(ahead - cfg.stop_gap_m + crossing_extra, cfg.brake_mps2, cfg.reaction_s)
                traffic_cap = min(traffic_cap, stop)
                if stop < cap:
                    cap, state, reason = stop, 'YIELD', 'mover in the path'
                self._note_pedestrian(o)
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
                stop = stopping_speed(ahead - cfg.stop_gap_m + crossing_extra, cfg.brake_mps2, cfg.reaction_s)
                traffic_cap = min(traffic_cap, stop)
                if stop < cap:
                    cap, state, reason = stop, 'YIELD', 'predicted crossing at s+%.1f m' % (
                        o['s'] - s_ego)
                # While Ego itself moves sideways (a pass), static roadside
                # clutter picks up lateral velocity; keep only the yield.
                self._note_pedestrian(o, binding=stop < speed and self.active is None)

        hold_cap = self._pedestrian_hold_cap(items, speed, s_ego, front_s, offset_at)
        if hold_cap is not None:
            traffic_cap = min(traffic_cap, hold_cap)
            if hold_cap <= cap:
                cap, state = hold_cap, 'YIELD'
                reason = 'holding for pedestrian at s+%.1f m' % (
                    self.pedestrian_hold['s'] - s_ego)
        self.person_d = {o['id']: (self.person_d.get(o['id'], ())[-1:] + (o['d'],))
                         for o in items if self._person_sized(o)}

        path = []
        s = s_ego
        while s <= s_ego + cfg.path_length_m and s <= self.route.s[-1]:
            xy, _ = self.route.pose(s, offset_at(s))
            path.append(xy)
            s += cfg.path_step_m
        return Decision(state, path, cap, reason, offset_now, items, traffic_cap)
