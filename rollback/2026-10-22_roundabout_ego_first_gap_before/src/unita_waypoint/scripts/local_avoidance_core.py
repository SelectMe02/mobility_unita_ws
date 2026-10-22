"""Map-frame obstacle tracking and temporary route planning (no ROS dependency).

The state flow follows the RoboRacer raceline/trailing/overtake/recovery idea.
All geometry and speed limits belong to this vehicle and its saved map.
"""
from dataclasses import dataclass
import math


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def project_polyline(point, points):
    """Nearest (distance, progress, tangent) on an open polyline."""
    best = None
    progress = 0.
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        t = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/length**2))
        projected = (a[0]+t*dx, a[1]+t*dy)
        item = (distance(point, projected), progress+t*length, (dx/length, dy/length))
        if best is None or item[0] < best[0]:
            best = item
        progress += length
    return best


def smoothstep(t):
    t = max(0., min(1., t))
    return t * t * (3. - 2. * t)


# Quintic minimum-jerk lane change (Werling et al., ICRA 2010): zero lateral
# speed and acceleration at both ends. Peak slope 1.875*d/L, peak curvature
# 5.774*d/L^2 (smoothstep jumps to 6*d/L^2 at its start and end).
QUINTIC_PEAK_SLOPE = 1.875
QUINTIC_PEAK_CURVATURE = 10./math.sqrt(3.)


def quintic(t):
    t = max(0., min(1., t))
    return t * t * t * (10. + t * (-15. + 6. * t))


def path_prefix(path, length_m):
    """Clip a polyline at an exact arc distance, including its end point."""
    if not path:
        return []
    result = [path[0]]
    remaining = length_m
    for a, b in zip(path, path[1:]):
        length = distance(a, b)
        if length <= 1e-9:
            continue
        if length >= remaining:
            t = remaining / length
            result.append((a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])))
            break
        result.append(b)
        remaining -= length
    return result


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    vx: float = 0.
    vy: float = 0.
    kind: str = 'unknown'
    confirmed: bool = True
    velocity_reliable: bool = True
    # Lane-aligned rectangle from the detector. Zero size means a legacy
    # circle input; it is then treated as the square enclosing the circle.
    length: float = 0.
    width: float = 0.
    yaw: float = 0.
    road_surface: bool = False
    track_id: int = 0
    raceline: int = 0
    lane_sector: int = 0
    lane_stable: bool = False
    lane_boundary_crossing: bool = True
    raw_width: float = 0.
    predicted: bool = False
    measurement_stamp: float = 0.

    def velocity(self):
        return (self.vx, self.vy) if self.velocity_reliable else (0., 0.)

    def corners(self):
        if self.length > 0. and self.width > 0.:
            half_l, half_w, yaw = self.length*.5, self.width*.5, self.yaw
        else:
            half_l = half_w = self.radius
            yaw = 0.
        c, s = math.cos(yaw), math.sin(yaw)
        return [(self.x+c*a-s*b, self.y+s*a+c*b)
                for a, b in ((half_l, half_w), (half_l, -half_w),
                             (-half_l, -half_w), (-half_l, half_w))]


@dataclass
class Decision:
    state: str
    path: list
    speed_cap_kmh: float
    obstacle_kind: str = ''
    obstacle_distance_m: float = float('inf')
    reason: str = ''


class LocalAvoidancePlanner:
    def __init__(self, route, sectors, lanes, *, half_width=0.946,
                 margin=0.35, front_extent=3.845, reaction_s=0.4,
                 decel=3.0, lookahead_m=60.,
                 obstacle_horizon_m=45., lane_indices=None, preferred_lane=1,
                 rear_extent=.790, lane_change_gap_m=3.,
                 lane_change_lateral_accel=1.5, lane_change_max_heading_deg=10.,
                 preferred_lane_indices=None, finish_line=None,
                 clear_hold_s=.4, slow_lead_margin_mps=1.,
                 avoidance_brake_limit=.35, restart_accel_rate=.5, route_lanes=None,
                 cancel_confirm_scans=3, cancel_max_progress=.6,
                 sector_min_speed_kmh=45., lane_change_speed_boost_kmh=10.,
                 min_speed_sectors=(11,), highway_sectors=(),
                 idm_time_gap_s=.2, idm_min_gap_m=3., idm_accel_mps2=1.5,
                 idm_comfort_decel_mps2=1.5, idm_command_horizon_s=.35,
                 mobil_politeness=.2, mobil_threshold_mps2=.2, mobil_safe_decel_mps2=4.,
                 highway_preferred_lane=3, highway_lane_bias_mps2=.6,
                 highway_obstacle_horizon_m=55., highway_change_hold_m=30.,
                 min_turn_radius_m=5.4, roundabout_conflict_point=None,
                 roundabout_approach_points=(), roundabout_sector=9,
                 roundabout_conflict_radius_m=3.5,
                 roundabout_monitor_distance_m=30.,
                 roundabout_approach_width_m=4.5,
                 roundabout_unknown_speed_kmh=20.,
                 roundabout_time_gap_s=2.5,
                 roundabout_clear_scans=3):
        self.route = list(route)
        self.sectors = list(sectors)
        # Raceline each route point was taken from (S11 may be driven on R3).
        self.route_lanes = list(route_lanes) if route_lanes is not None else [1]*len(self.route)
        if len(self.route_lanes) != len(self.route):
            raise ValueError('route lane array must match the route')
        self.lanes = {int(k): list(v) for k, v in lanes.items()}
        self.lane_indices = lane_indices
        self.half_width = half_width
        self.margin = margin
        self.front_extent = front_extent
        self.reaction_s = reaction_s
        self.decel = decel
        self.lookahead_m = lookahead_m
        self.obstacle_horizon_m = obstacle_horizon_m
        self.obstacle_horizon_path = []
        self.preferred_lane = int(preferred_lane)
        self.preferred_lane_indices = preferred_lane_indices
        self.finish_line = finish_line
        self.finish_line_rear_y = None
        self.finish_line_passed = None
        self.finish_priority_active = False
        self.preference_ends = []
        if preferred_lane_indices is not None:
            for i, lane in sorted(preferred_lane_indices.items()):
                after = preferred_lane_indices.get(i+1, self.preferred_lane)
                if lane != after and i+1 < len(self.route):
                    start = i
                    while preferred_lane_indices.get(start-1) == lane:
                        start -= 1
                    self.preference_ends.append((start, i, lane, after))
        self.rear_extent = rear_extent
        # Bumper-to-rectangle gap required ahead and behind in every lane a
        # lane change enters. NPC vehicles keep their lane, so no speed term.
        self.lane_change_gap_m = lane_change_gap_m
        # A leader this much slower than the desired speed is worth passing.
        self.slow_lead_margin_mps = slow_lead_margin_mps
        # Lane-change length: whichever is longer of a lateral-acceleration
        # limit (fast) and a heading-angle limit (slow), so the car follows
        # the curve without crossing the next lane line.
        self.lane_change_lateral_accel = lane_change_lateral_accel
        self.lane_change_max_heading_deg = lane_change_max_heading_deg
        # Once settled in the preferred lane of a priority interval, only the
        # neighbouring lane is used to pass, and changes are never cancelled.
        self.lane_hold = None
        self.current_lane = 1
        self.effective_preferred_lane = 1
        self.preference_reason = ''
        self.previous_preferred_lane = None
        self.transition = None
        self.departed_preferred = False
        self.previous_arc = None
        self.lap_offset = 0.
        if len(self.route) != len(self.sectors) or len(self.route) < 2:
            raise ValueError('route and sector arrays must match')
        self.arc = [0.]
        for a, b in zip(self.route, self.route[1:]):
            self.arc.append(self.arc[-1]+distance(a, b))
        self.total_length = self.arc[-1]+distance(self.route[-1], self.route[0])
        self.active_lane = None
        self.clear_hold_s = clear_hold_s
        self.avoidance_brake_limit = avoidance_brake_limit
        self.restart_accel_rate = restart_accel_rate
        self.gap_blocker = None
        # A lane change in progress is cancelled only for a gap blocker in the
        # target lane that persists this many plans, is closing in, and only
        # before this fraction of the change; otherwise it is completed.
        self.cancel_confirm_scans = int(cancel_confirm_scans)
        self.cancel_max_progress = cancel_max_progress
        self.cancel_votes = 0
        # Followed by the controller in min_speed_sectors: no standstill
        # unless something is directly ahead, and accelerate while changing.
        self.sector_min_speed_kmh = sector_min_speed_kmh
        self.lane_change_speed_boost_kmh = lane_change_speed_boost_kmh
        self.min_speed_sectors = tuple(min_speed_sectors)
        self.lane_end_imminent = False
        # Highway sectors: compact bumper-gap/closing-speed following,
        # MOBIL-style lane selection with a bias toward the preferred
        # lane, quintic lane changes whose length adapts to the free space
        # instead of a speed precondition, and no planner standstill.
        self.highway_sectors = tuple(int(s) for s in highway_sectors)
        self.idm_time_gap_s = idm_time_gap_s
        self.idm_min_gap_m = idm_min_gap_m
        self.idm_accel_mps2 = idm_accel_mps2
        self.idm_comfort_decel_mps2 = idm_comfort_decel_mps2
        self.idm_command_horizon_s = idm_command_horizon_s
        self.mobil_politeness = mobil_politeness
        self.mobil_threshold_mps2 = mobil_threshold_mps2
        self.mobil_safe_decel_mps2 = mobil_safe_decel_mps2
        self.highway_preferred_lane = int(highway_preferred_lane)
        self.highway_lane_bias_mps2 = highway_lane_bias_mps2
        self.highway_obstacle_horizon_m = highway_obstacle_horizon_m
        self.highway_change_hold_m = highway_change_hold_m
        self.highway_change_end = None
        self.highway_active = False
        self.highway_emergency = False
        self.highway_road_returns_ignored = 0
        # Tightest path the car can steer (wheelbase / tan(max steering)):
        # the shortest, slow merge out of a lane that is about to end.
        self.min_turn_radius_m = min_turn_radius_m
        self.roundabout_conflict_point = (None if roundabout_conflict_point is None else
                                          tuple(map(float, roundabout_conflict_point)))
        self.roundabout_approach_points = [tuple(map(float, p))
                                           for p in roundabout_approach_points]
        self.roundabout_sector = int(roundabout_sector)
        self.roundabout_conflict_radius_m = float(roundabout_conflict_radius_m)
        self.roundabout_monitor_distance_m = float(roundabout_monitor_distance_m)
        self.roundabout_approach_width_m = float(roundabout_approach_width_m)
        self.roundabout_unknown_speed_mps = float(roundabout_unknown_speed_kmh)/3.6
        self.roundabout_time_gap_s = float(roundabout_time_gap_s)
        self.roundabout_clear_scans = int(roundabout_clear_scans)
        self.roundabout_clear_votes = 0
        self.roundabout_last_clear_stamp = None
        self.roundabout_state = 'OUTSIDE'
        self.roundabout_threat_id = None
        self.roundabout_ego_eta_s = None
        self.roundabout_actor_eta_s = None
        self.roundabout_yield_seen = False
        self.roundabout_gap_accepted = False
        self.roundabout_conflict_arc = None
        self.roundabout_actor_conflict_progress = None
        if self.roundabout_conflict_point is not None:
            if len(self.roundabout_conflict_point) != 2:
                raise ValueError('roundabout conflict point must be XY')
            conflict_index = min(range(len(self.route)),
                                 key=lambda i: distance(self.route[i], self.roundabout_conflict_point))
            tangent = self._tangent(conflict_index)
            base = self.route[conflict_index]
            self.roundabout_conflict_arc = (self.arc[conflict_index]
                +(self.roundabout_conflict_point[0]-base[0])*tangent[0]
                +(self.roundabout_conflict_point[1]-base[1])*tangent[1])
        if self.roundabout_approach_points:
            projected = project_polyline(self.roundabout_conflict_point,
                                         self.roundabout_approach_points)
            self.roundabout_actor_conflict_progress = projected[1] if projected else None
        if min(idm_time_gap_s, idm_min_gap_m, idm_accel_mps2, idm_comfort_decel_mps2,
               idm_command_horizon_s, mobil_threshold_mps2, mobil_safe_decel_mps2,
               highway_obstacle_horizon_m, min_turn_radius_m) <= 0 or min(
                   mobil_politeness, highway_lane_bias_mps2, highway_change_hold_m) < 0:
            raise ValueError('invalid highway IDM/MOBIL settings')
        if min(half_width, margin, front_extent, reaction_s, decel,
               lookahead_m, obstacle_horizon_m, rear_extent,
               lane_change_gap_m, lane_change_lateral_accel, lane_change_max_heading_deg,
               slow_lead_margin_mps,
               clear_hold_s, avoidance_brake_limit, restart_accel_rate) <= 0 or avoidance_brake_limit > 1. or self.preferred_lane not in (1, 2, 3):
            raise ValueError('invalid avoidance geometry')
        if self.roundabout_conflict_point is not None and (
                len(self.roundabout_approach_points) < 2
                or min(self.roundabout_conflict_radius_m,
                       self.roundabout_monitor_distance_m,
                       self.roundabout_approach_width_m,
                       self.roundabout_unknown_speed_mps,
                       self.roundabout_time_gap_s) <= 0
                or self.roundabout_clear_scans < 1):
            raise ValueError('invalid roundabout conflict geometry')

    def _forward(self, index, distance_m):
        result = [(index, self.route[index], 0.)]
        count = len(self.route)
        travelled = 0.
        for step in range(1, count):
            j = (index + step) % count
            previous = result[-1][1]
            travelled += distance(previous, self.route[j])
            if travelled > distance_m:
                break
            result.append((j, self.route[j], travelled))
        return result

    def _lane_at(self, number, point, index):
        # Saved adjacent lanes cover only part of sector 11. Never extrapolate.
        if self.lane_indices is not None:
            # Saved lanes are mapped to route progress; an unmapped index is
            # no lane (e.g. Raceline 1 after its lane drop), never a projection
            # onto the joined polyline of all sectors.
            return self.lane_indices.get(number, {}).get(index)
        lane = self.lanes.get(number, [])
        if not lane:
            return None
        closest = min(lane, key=lambda p: distance(p, point))
        # Without source IDs, require a perpendicular projection inside a segment.
        nearest_index = lane.index(closest)
        segments = [(lane[i], lane[i+1]) for i in
                    range(max(0, nearest_index-1), min(len(lane)-1, nearest_index+1))]
        valid = []
        for start, end in segments:
            dx, dy = end[0]-start[0], end[1]-start[1]
            squared = dx*dx + dy*dy
            if squared < 1e-6:
                continue
            t = ((point[0]-start[0])*dx+(point[1]-start[1])*dy)/squared
            if -.001 <= t <= 1.001:
                valid.append((start[0]+t*dx, start[1]+t*dy))
        if not valid:
            return None
        return min(valid, key=lambda p: distance(p, point))

    def _tangent(self, index):
        a, b = self.route[index], self.route[(index+1) % len(self.route)]
        norm = max(distance(a, b), 1e-6)
        return ((b[0]-a[0])/norm, (b[1]-a[1])/norm)

    def _progress_at(self, index, ego_xy):
        raw = self.arc[index]
        if self.previous_arc is not None:
            delta = raw-self.previous_arc
            if delta < -self.total_length/2.:
                self.lap_offset += self.total_length
            elif delta > self.total_length/2.:
                self.lap_offset -= self.total_length
        self.previous_arc = raw
        tangent = self._tangent(index)
        point = self.route[index]
        return raw+self.lap_offset + ((ego_xy[0]-point[0])*tangent[0] +
                                     (ego_xy[1]-point[1])*tangent[1])

    def _lane_point(self, lane, point, index):
        line = self.finish_line
        if (line and line['joins_primary'] and lane == line['lane'] and
                line['end_index'] < index and
                self.arc[index]-self.arc[line['end_index']] <= self.lookahead_m+10.):
            # The saved R3 joins R1 at CP11. Follow the existing R1 pavement
            # through the join while the rear of the car clears the line.
            return point
        return point if lane == self.route_lanes[index] else self._lane_at(lane, point, index)

    def _span(self, lane, ahead):
        end = 0.
        for i, point, s in ahead:
            if self._lane_point(lane, point, i) is None:
                break
            end = s
        return end

    def _requested_lane(self, index, ego_xy, ego_yaw=None):
        self.finish_priority_active = False
        if self.finish_line:
            line = self.finish_line
            if ego_yaw is None:
                tx, ty = self._tangent(index)
                ego_yaw = math.atan2(ty, tx)
            # Highest Y among the four vehicle corners. For southbound
            # travel this is the trailing rear corner, including yaw/width.
            sy, cy = math.sin(ego_yaw), math.cos(ego_yaw)
            self.finish_line_rear_y = ego_xy[1]+max(self.front_extent*sy, -self.rear_extent*sy)+self.half_width*abs(cy)
            self.finish_line_passed = self.finish_line_rear_y < line['y']
            # R1 indexing can enter S12 while the rear is still in S11.
            # The exit decision itself depends only on the body Y envelope.
            near_exit = self.sectors[index] == line['sector']+1
            inside = (index >= line['start_index'] and
                      (self.sectors[index] == line['sector'] or near_exit))
            self.finish_priority_active = inside and not self.finish_line_passed
            if not inside:
                self.finish_line_rear_y = self.finish_line_passed = None
            return line['lane'] if self.finish_priority_active else self.preferred_lane
        if self.preferred_lane_indices is None:
            return self.preferred_lane
        requested = self.preferred_lane_indices.get(index, self.preferred_lane)
        for start, end, lane, after in self.preference_ends:
            if index < start or self.sectors[index] != self.sectors[end]:
                continue
            # These checkpoint intervals run south: crossing is strictly a
            # decrease below the checkpoint Y, regardless of X or the nearest
            # R1 sample. Equality still belongs to the preferred interval.
            crossed = ego_xy[1] < self.route[end][1]
            return after if crossed else lane
        return requested

    def _merge_length(self, start, end, speed_mps=0.):
        """Smoothstep lane change: peak curvature 6d/L^2, peak slope 1.5d/L."""
        lateral = max(.1, distance(start, end))
        by_accel = max(0., speed_mps)*math.sqrt(6.*lateral/self.lane_change_lateral_accel)
        heading = math.radians(min(max(self.lane_change_max_heading_deg, 1.), 45.))
        return max(by_accel, 1.5*lateral/math.tan(heading))

    def _path(self, ahead, ego_xy, lane, progress, transition=None):
        # The transition has a fixed route anchor. Replanning does not move
        # its end forward, and its first point always matches the actual ego.
        target0 = self._lane_point(lane, ahead[0][1], ahead[0][0])
        if target0 is None:
            return []
        blend_of = (quintic if transition is not None and transition.get('shape') == 'quintic'
                    else smoothstep)
        if transition is None:
            correction = (ego_xy[0]-target0[0], ego_xy[1]-target0[1])
        else:
            origin0 = self._lane_point(transition['source'], ahead[0][1], ahead[0][0])
            if origin0 is None:
                return []
            t = blend_of((progress-transition['start'])/transition['length'])
            expected = (origin0[0]+t*(target0[0]-origin0[0]),
                        origin0[1]+t*(target0[1]-origin0[1]))
            correction = (ego_xy[0]-expected[0], ego_xy[1]-expected[1])
        path = []
        for i, point, s in ahead:
            target = self._lane_point(lane, point, i)
            if target is None:
                break
            if transition is not None:
                blend = blend_of((progress+s-transition['start'])/transition['length'])
                origin = self._lane_point(transition['source'], point, i)
                if origin is None and blend < 1.:
                    break
                if origin is not None:
                    target = (origin[0]+blend*(target[0]-origin[0]),
                              origin[1]+blend*(target[1]-origin[1]))
            correction_length = (max(8., transition['length']*.5)
                                 if transition is not None else 8.)
            decay = 1.-smoothstep(s/correction_length)
            path.append((target[0]+correction[0]*decay,
                         target[1]+correction[1]*decay))
        return path

    def _lead(self, path, ego_xy, obstacles, horizon_m=None, lane=None):
        """Nearest obstacle whose rectangle overlaps the ego footprint band."""
        horizon_m = self.obstacle_horizon_m if horizon_m is None else horizon_m
        if len(path) < 2:
            return None
        dx, dy = path[1][0]-path[0][0], path[1][1]-path[0][1]
        norm = max(math.hypot(dx, dy), 1e-6)
        travelled, samples = 0., []
        for a, b in zip(path, path[1:]):
            samples.append((a, b, travelled))
            travelled += distance(a, b)
        band = self.half_width+self.margin
        found = []
        for obstacle in obstacles:
            # Behind us is relevant to merging, never a leader to tail.
            if (obstacle.x-ego_xy[0])*dx/norm + (obstacle.y-ego_xy[1])*dy/norm < 0.:
                continue
            projections = []
            for a, b, start in samples:
                bx, by = b[0]-a[0], b[1]-a[1]
                length = distance(a, b)
                if length < 1e-6:
                    continue
                t = max(0., min(1., ((obstacle.x-a[0])*bx +
                                     (obstacle.y-a[1])*by)/length**2))
                p = (a[0]+t*bx, a[1]+t*by)
                projections.append((distance(p, (obstacle.x, obstacle.y)),
                                    start+t*length, p, (bx/length, by/length)))
            if not projections:
                continue
            _, s, point, tangent = min(projections, key=lambda item: item[0])
            longitudinal, lateral = [], []
            for cx, cy in obstacle.corners():
                ox, oy = cx-point[0], cy-point[1]
                longitudinal.append(ox*tangent[0]+oy*tangent[1])
                lateral.append(-ox*tangent[1]+oy*tangent[0])
            if s > horizon_m or min(lateral) > band or max(lateral) < -band:
                continue
            if (self.highway_active and self.transition is None and lane is not None
                    and obstacle.lane_stable and obstacle.lane_sector == 11
                    and obstacle.raceline in (1, 2, 3) and obstacle.raceline != lane
                    and not obstacle.lane_boundary_crossing
                    and (min(lateral) > self.half_width
                         or max(lateral) < -self.half_width)):
                # A stable neighbouring car wholly inside its own lane is
                # not our leader just because it touches the extra margin.
                # Keep measured-body overlap protection, even across labels.
                continue
            clearance = max(0., s+min(longitudinal))
            vx, vy = obstacle.velocity()
            if self.highway_active and obstacle.confirmed and not obstacle.velocity_reliable:
                vx, vy = obstacle.vx, obstacle.vy
            velocity = max(0., vx*tangent[0]+vy*tangent[1])
            found.append((clearance, obstacle, velocity))
        return min(found, key=lambda item: item[0]) if found else None

    def _slow(self, lead, desired_mps):
        return lead is not None and lead[2] < desired_mps-self.slow_lead_margin_mps

    def _gap_blocker(self, index, ego_xy, available, destination, obstacles):
        """First obstacle within lane_change_gap_m of the ego bumpers in any
        lane between the ego's lateral position and the destination lane."""
        tangent = self._tangent(index)
        normal = (-tangent[1], tangent[0])
        def lateral_of(point):
            return (point[0]-ego_xy[0])*normal[0]+(point[1]-ego_xy[1])*normal[1]
        goal = lateral_of(available[destination])
        lanes = [lateral_of(point) for n, point in available.items()
                 if n == destination or (abs(lateral_of(point)) > .3 and
                                         lateral_of(point)*goal > 0. and
                                         abs(lateral_of(point)) < abs(goal))]
        band = self.half_width+self.margin
        for obstacle in obstacles:
            longitudinal, lateral = [], []
            for cx, cy in obstacle.corners():
                ox, oy = cx-ego_xy[0], cy-ego_xy[1]
                longitudinal.append(ox*tangent[0]+oy*tangent[1])
                lateral.append(ox*normal[0]+oy*normal[1])
            if not any(min(lateral) <= c+band and max(lateral) >= c-band for c in lanes):
                continue
            front_gap = min(longitudinal)-self.front_extent
            rear_gap = -self.rear_extent-max(longitudinal)
            if front_gap < self.lane_change_gap_m and rear_gap < self.lane_change_gap_m:
                return obstacle, front_gap
        return None

    def _closing(self, index, ego_xy, speed_mps, obstacle):
        """True if the obstacle is beside the ego or the gap to it is shrinking."""
        tangent = self._tangent(index)
        along = (obstacle.x-ego_xy[0])*tangent[0]+(obstacle.y-ego_xy[1])*tangent[1]
        vx, vy = obstacle.velocity()
        speed = vx*tangent[0]+vy*tangent[1]
        if along > self.front_extent:
            return speed < speed_mps-.5   # ahead and slower
        if along < -self.rear_extent:
            return speed > speed_mps+.5   # behind and faster
        return True                       # alongside

    def _gap_threat(self, blocker):
        # Only a blocker ahead of the front bumper limits speed.
        obstacle, front_gap = blocker
        if front_gap < -self.front_extent:
            return None
        return (max(0., front_gap)+self.front_extent, obstacle, 0.)

    def _stop_cap(self, clearance, lead_speed=0.):
        bumper_gap = clearance-self.front_extent-self.margin
        if bumper_gap <= 0.:
            return 0.
        available = max(0., bumper_gap-lead_speed*1.2)
        safe = math.sqrt((self.decel*self.reaction_s)**2 +
                         2.*self.decel*available)-self.decel*self.reaction_s
        return max(0., (lead_speed+safe)*3.6)

    def _roundabout_guard(self, index, progress, ego_speed_mps, desired_speed_mps,
                          current_path, obstacles, edge_cap, observation_stamp):
        """Yield at the S9 merge for traffic approaching on the circulating lane.

        This is deliberately map- and sector-scoped. Ordinary adjacent traffic
        elsewhere continues through the existing path-overlap logic.
        """
        if (self.roundabout_conflict_arc is None
                or self.sectors[index] != self.roundabout_sector):
            self.roundabout_state = 'OUTSIDE'
            self.roundabout_clear_votes = 0
            self.roundabout_last_clear_stamp = None
            self.roundabout_threat_id = None
            self.roundabout_ego_eta_s = self.roundabout_actor_eta_s = None
            self.roundabout_yield_seen = False
            self.roundabout_gap_accepted = False
            return None
        conflict_progress = self.roundabout_conflict_arc+self.lap_offset
        to_conflict = conflict_progress-progress
        if to_conflict < -self.total_length/2.:
            to_conflict += self.total_length
        elif to_conflict > self.total_length/2.:
            to_conflict -= self.total_length
        if to_conflict > self.roundabout_monitor_distance_m:
            self.roundabout_state = 'OUTSIDE'
            self.roundabout_clear_votes = 0
            self.roundabout_last_clear_stamp = None
            self.roundabout_yield_seen = False
            self.roundabout_gap_accepted = False
            return None
        # The front of Ego has crossed the near edge of the merge area. Avoid
        # an abrupt stop beside a circulating car; normal path collision
        # protection remains active from this point onward.
        if to_conflict <= self.front_extent+self.roundabout_conflict_radius_m:
            self.roundabout_state = 'COMMITTED'
            self.roundabout_clear_votes = self.roundabout_clear_scans
            self.roundabout_threat_id = None
            self.roundabout_ego_eta_s = self.roundabout_actor_eta_s = None
            return None

        # Once Ego has actually yielded to a circulating car, stopped, and
        # observed a clear gap for the requested scans, preserve that accepted
        # gap until the front enters the merge. Otherwise a following car first
        # appearing during pedal-release delay can cancel every departure.
        if self.roundabout_gap_accepted:
            self.roundabout_state = 'ENTRY_CLEAR'
            self.roundabout_threat_id = None
            self.roundabout_ego_eta_s = self.roundabout_actor_eta_s = None
            return self._decision('ROUNDABOUT_ENTRY', current_path, edge_cap,
                                  reason='roundabout_gap_accepted')

        self.roundabout_state = 'APPROACH'
        travel_speed = max(1., ego_speed_mps,
                           0. if desired_speed_mps is None else desired_speed_mps)
        ego_eta = max(0., to_conflict-self.front_extent)/travel_speed
        threat = None
        threat_eta = None
        for obstacle in obstacles:
            if obstacle.road_surface:
                continue
            projection = project_polyline((obstacle.x, obstacle.y),
                                          self.roundabout_approach_points)
            if projection is None:
                continue
            lateral, actor_progress, tangent = projection
            if lateral > self.roundabout_approach_width_m*.5+obstacle.radius:
                continue
            remaining = self.roundabout_actor_conflict_progress-actor_progress
            if remaining < -self.roundabout_conflict_radius_m-obstacle.radius:
                continue                 # already clear of the merge
            if obstacle.velocity_reliable:
                actor_speed = obstacle.vx*tangent[0]+obstacle.vy*tangent[1]
                if actor_speed <= .5:
                    # A stable object before the merge is not circulating
                    # traffic. A vehicle already in the merge still blocks it.
                    if remaining > self.roundabout_conflict_radius_m+obstacle.radius:
                        continue
                    actor_eta = 0.
                else:
                    actor_eta = max(0., remaining-self.roundabout_conflict_radius_m
                                    -obstacle.radius)/actor_speed
            else:
                # A new left-side track gets the scenario's circulating speed
                # until the detector has enough history for a reliable vector.
                actor_eta = max(0., remaining-self.roundabout_conflict_radius_m
                                -obstacle.radius)/self.roundabout_unknown_speed_mps
            if abs(actor_eta-ego_eta) > self.roundabout_time_gap_s:
                continue
            if threat is None or actor_eta < threat_eta:
                threat, threat_eta = obstacle, actor_eta

        self.roundabout_ego_eta_s = ego_eta
        self.roundabout_actor_eta_s = threat_eta
        if threat is not None:
            self.roundabout_state = 'YIELD'
            self.roundabout_yield_seen = True
            self.roundabout_clear_votes = 0
            self.roundabout_last_clear_stamp = observation_stamp
            self.roundabout_threat_id = threat.track_id
            clearance = max(0., to_conflict-self.roundabout_conflict_radius_m)
            cap = min(edge_cap, self._stop_cap(clearance))
            return self._decision('ROUNDABOUT_YIELD', current_path, cap,
                                  (clearance, threat, 0.), 'roundabout_conflict')

        self.roundabout_threat_id = None
        distinct_scan = (observation_stamp is None
                         or observation_stamp != self.roundabout_last_clear_stamp)
        if distinct_scan:
            self.roundabout_clear_votes = min(self.roundabout_clear_scans,
                                               self.roundabout_clear_votes+1)
            self.roundabout_last_clear_stamp = observation_stamp
        if self.roundabout_clear_votes >= self.roundabout_clear_scans:
            self.roundabout_state = 'ENTRY_CLEAR'
            if self.roundabout_yield_seen and ego_speed_mps <= .5:
                self.roundabout_gap_accepted = True
                return self._decision('ROUNDABOUT_ENTRY', current_path, edge_cap,
                                      reason='roundabout_gap_accepted')
            return None
        # This only has an effect if the car reaches the entry before the
        # requested consecutive clear scans have arrived.
        clearance = max(0., to_conflict-self.roundabout_conflict_radius_m)
        cap = min(edge_cap, self._stop_cap(clearance))
        return self._decision('ROUNDABOUT_APPROACH', current_path, cap,
                              reason='roundabout_clear_confirmation')

    def _decision(self, state, path, cap, threat=None, reason=''):
        return Decision(state, path, cap, threat[1].kind if threat else '',
                        threat[0] if threat else float('inf'), reason)

    def _lane_end_cap(self, ahead, available, preferred, base, speed_mps, merge_length=None):
        merge_length = merge_length or self._merge_length
        edge_cap = float('inf')
        self.lane_end_imminent = False
        span = self._span(self.current_lane, ahead)
        if span < ahead[-1][2]:
            # The lane being driven ends ahead (saved side lane, or a lane drop
            # on the route itself). Merge into the lane that continues.
            changing = self.transition['target'] if self.transition is not None else None
            target = next((n for n in (changing, preferred, base)
                           if n is not None and n != self.current_lane and n in available
                           and self._span(n, ahead) > span), None)
            if target is not None:
                if changing == target:
                    # Already merging: stop the bumper short of the lane end.
                    edge_cap = self._stop_cap(span)
                    self.lane_end_imminent = span < speed_mps*self.reaction_s+self.front_extent+8.
                else:
                    # Stop where a low-speed (heading-limited) merge still fits.
                    room = span - merge_length(
                        available[self.current_lane], available[target], 0.) - 8.
                    edge_cap = (self._stop_cap(room+self.front_extent+self.margin)
                                if room > 0. else 0.)
                    self.lane_end_imminent = room <= 0.
                if self.finish_priority_active and self.current_lane != base:
                    edge_cap = min(edge_cap, self._stop_cap(span))
        return edge_cap

    def plan(self, index, ego_xy, speed_mps, obstacles, ego_yaw=None,
             desired_speed_mps=None, observation_stamp=None):
        # See lane endings early enough to decelerate before merging back.
        # Long enough to plan a two-lane change and see the lane ending.
        # A lane drop must be seen early enough to merge, or else brake to a
        # stop before the lane ends.
        horizon = max(self.lookahead_m, self.obstacle_horizon_m+10.,
                      speed_mps**2/(2.*self.decel)+40.,
                      self._merge_length((0., 0.), (0., 8.), speed_mps)
                      + speed_mps**2/(2.*self.decel) + speed_mps*self.reaction_s + 20.)
        self.obstacle_horizon_path = []
        self.gap_blocker = None
        # Far, tiny single-frame returns must mature before selecting a new
        # lane. Keep ALL objects within the stopping envelope, and all larger
        # objects, so an unconfirmed near obstacle still triggers protection.
        stopping = speed_mps**2/(2.*self.decel)+speed_mps*self.reaction_s
        near_distance = self.front_extent+max(6., stopping)+self.margin
        obstacles = [o for o in obstacles if o.confirmed or o.radius >= .35
                     or distance(ego_xy, (o.x, o.y))-o.radius <= near_distance]
        ahead = self._forward(index, horizon)
        if len(ahead) < 2:
            return self._decision('BLOCKED', [], 0., reason='saved_path_ends')
        progress = self._progress_at(index, ego_xy)
        base = self.route_lanes[index]
        available = {n: self._lane_point(n, ahead[0][1], index)
                     for n in sorted({1, 2, 3} | set(self.lanes))}
        available = {n: point for n, point in available.items() if point is not None}
        self.current_lane = min(available, key=lambda n: distance(ego_xy, available[n]))
        requested = self._requested_lane(index, ego_xy, ego_yaw)
        self.highway_active = self.sectors[index] in self.highway_sectors
        if self.highway_active:
            return self._plan_highway(index, ego_xy, speed_mps, obstacles,
                                      desired_speed_mps, ahead, progress, available)
        self.preference_reason = 'priority_interval' if requested != self.preferred_lane else 'default_lane'
        preferred = requested if requested in available else base
        if requested not in available:
            self.preference_reason = 'saved_lane_unavailable'
        if self.finish_priority_active:
            # Missing/short geometry must cause slowing or holding, never
            # silently change the user's R3 policy before the finish line.
            preferred = requested
            if preferred not in available:
                self.effective_preferred_lane = preferred
                return self._decision('BLOCKED', [], 0., reason='preferred_saved_path_missing')
            if distance(available[preferred], available[base]) < .35:
                self.current_lane = preferred
        if preferred != base and not self.finish_priority_active:
            exit_length = self._merge_length(available[preferred], available[base], speed_mps)
            if self._span(preferred, ahead) < exit_length+8.:
                preferred = base
                self.preference_reason = 'saved_lane_ending'
        self.effective_preferred_lane = preferred
        if (self.previous_preferred_lane is not None and
                preferred != self.previous_preferred_lane and self.current_lane != preferred):
            self.departed_preferred = True
        self.previous_preferred_lane = preferred
        if self.preference_reason != 'priority_interval' or preferred != requested:
            self.lane_hold = None
        elif self.lane_hold is None and self.transition is None and self.current_lane == preferred:
            self.lane_hold = preferred
        edge_cap = self._lane_end_cap(ahead, available, preferred, base, speed_mps)
        current_path = self._path(ahead, ego_xy, self.current_lane, progress)
        self.obstacle_horizon_path = path_prefix(current_path, self.obstacle_horizon_m)
        # Without a speed target every leader counts as slow (pass if possible).
        desired = float('inf') if desired_speed_mps is None else desired_speed_mps
        if self.transition is not None:
            decision = self._transition_step(index, ego_xy, speed_mps, progress, ahead,
                                             available, obstacles, edge_cap)
            if decision is not None:
                return decision

        current = self.current_lane
        current_path = self._path(ahead, ego_xy, current, progress)
        roundabout = self._roundabout_guard(index, progress, speed_mps,
                                             desired_speed_mps, current_path,
                                             obstacles, edge_cap, observation_stamp)
        if roundabout is not None:
            self.obstacle_horizon_path = path_prefix(current_path, self.obstacle_horizon_m)
            return roundabout
        threat = self._lead(current_path, ego_xy, obstacles)
        candidates = []
        if current != preferred:
            candidates.append(preferred)
        if self._slow(threat, desired):
            candidates.extend(sorted(available, key=lambda n: distance(available[n], ego_xy)))
        if self.lane_hold is not None:
            # Held lane: pass only through the adjacent lane, then come back.
            candidates = [n for n in candidates if abs(n-self.lane_hold) <= 1
                          and (current == self.lane_hold or n == self.lane_hold)]
        # One lane at a time (R1 -> R2 -> R3), each step gap-checked anew.
        steps = {}
        for goal in candidates:
            if goal != current:
                steps.setdefault(current + (1 if goal > current else -1), goal)
        waiting = ''
        for lane, goal in steps.items():
            if lane not in available:
                continue
            length = self._merge_length(ego_xy, available[lane], speed_mps)
            if self._span(lane, ahead) < length+8. or self._span(current, ahead) < length:
                waiting = waiting or 'saved_lane_too_short'
                continue
            blocker = self._gap_blocker(index, ego_xy, available, lane, obstacles)
            if blocker is not None:
                self.gap_blocker = blocker
                waiting = 'lane_change_gap'
                continue
            # Moving traffic ahead in the new lane does not matter, unless it
            # is slow and no further away than the leader we already have.
            lead = self._lead(self._path(ahead, ego_xy, lane, progress), ego_xy, obstacles)
            # Passing through to a further goal lane only needs room for the
            # next step before reaching this lane's leader.
            through = (goal != lane and goal in available and lead is not None and
                       lead[0] > self._merge_length(available[lane], available[goal], speed_mps)
                       + self.lane_change_gap_m + self.front_extent)
            if (self._slow(lead, desired) and not through
                    and (threat is None or lead[0] <= threat[0]+5.)):
                waiting = waiting or 'target_lane_slow_vehicle'
                continue
            transition = dict(source=current, target=lane, start=progress,
                              length=length, state='', locked=self.lane_hold is not None)
            path = self._path(ahead, ego_xy, lane, progress, transition)
            transit = self._lead(path, ego_xy, obstacles)
            transit_cap = float('inf') if transit is None else self._stop_cap(transit[0], transit[2])
            cap = min(edge_cap, transit_cap)
            if transit_cap < speed_mps*3.6-3.6:
                # Too close to swerve at this speed: brake in lane, retry later.
                waiting = waiting or 'too_close_to_change'
                continue
            # Named after the step's final goal: R1 -> R2 on the way to a
            # preferred R3 is a lane change, not an overtake.
            state = ('RETURNING' if goal == preferred and self.departed_preferred
                     else 'OVERTAKING' if threat is not None or goal != preferred
                     else 'LANE_CHANGING')
            transition['state'] = state
            self.transition = transition
            self.departed_preferred = goal != preferred
            self.active_lane = lane if lane != base else None
            self.obstacle_horizon_path = path_prefix(path, self.obstacle_horizon_m)
            return self._decision(state, path, cap, transit or threat)

        # Stay in the current lane, following its leader if any.
        self.active_lane = current if current != base else None
        if current == preferred:
            self.departed_preferred = False
        if len(current_path) < 2:
            return self._decision('BLOCKED', current_path, 0., reason=waiting or 'saved_path_ends')
        path = current_path if current != base or distance(ego_xy, available[base]) > .35 else []
        if threat is not None:
            cap = min(edge_cap, self._stop_cap(threat[0], threat[2]))
            return self._decision('TAILING', path, cap, threat, waiting)
        state = 'LANE_WAIT' if waiting else 'OVERTAKING' if self.departed_preferred else 'RACELINE'
        return self._decision(state, path, edge_cap, reason=waiting)

    def _transition_step(self, index, ego_xy, speed_mps, progress, ahead,
                         available, obstacles, edge_cap):
        """Continue, finish or cancel the active lane change. Returns None
        when the change is complete and normal lane selection should run."""
        transition = self.transition
        lane = transition['target']
        if lane not in available or progress >= transition['start']+transition['length']:
            self.transition = None
            if lane in available:
                self.current_lane = lane
            return None
        reason = ''
        blocker = self._gap_blocker(index, ego_xy, available, lane, obstacles)
        fraction = (progress-transition['start'])/max(transition['length'], 1e-6)
        if blocker is not None and transition['state'] != 'CANCELLING':
            # Conservative cancel: the blocker must persist, be closing in, and
            # the change must still be in its first part. Otherwise finish it.
            closing = self._closing(index, ego_xy, speed_mps, blocker[0])
            self.cancel_votes = self.cancel_votes+1 if closing else 0
            if (self.cancel_votes < self.cancel_confirm_scans
                    or fraction > self.cancel_max_progress):
                self.gap_blocker = blocker
                reason = 'lane_change_gap_hold'
                blocker = None if not closing else blocker
                transition = dict(transition, locked=True)
        else:
            self.cancel_votes = 0
        if blocker is not None and (transition.get('locked')
                                    or transition['state'] == 'CANCELLING'):
            # No cancel in the held lane (or of a cancel): finish the change,
            # slowing if needed.
            self.gap_blocker = blocker
            reason = 'lane_change_gap_hold'
        elif blocker is not None:
            self.gap_blocker = blocker
            source = transition['source']
            back = (None if source not in available else
                    self._gap_blocker(index, ego_xy, available, source, obstacles))
            if source in available and back is None:
                # Someone entered the 3 m gap: steer back to the lane we left.
                length = self._merge_length(ego_xy, available[source], speed_mps)
                self.transition = transition = dict(
                    source=lane, target=source, start=progress, length=length,
                    state='CANCELLING')
                self.departed_preferred = source != self.effective_preferred_lane
                lane = source
                reason = 'lane_change_cancelled'
            else:
                # Both sides are occupied: keep the path and slow for anything ahead.
                reason = 'lane_change_gap_both_sides'
        path = self._path(ahead, ego_xy, lane, progress, transition)
        if len(path) < 2:
            return self._decision('BLOCKED', path, 0., reason='transition_path_unavailable')
        self.active_lane = lane if lane != self.route_lanes[index] else None
        self.obstacle_horizon_path = path_prefix(path, self.obstacle_horizon_m)
        threat = self._lead(path, ego_xy, obstacles)
        if blocker is not None and reason in ('lane_change_gap_both_sides', 'lane_change_gap_hold'):
            gap_threat = self._gap_threat(blocker)
            if gap_threat is not None and (threat is None or gap_threat[0] < threat[0]):
                threat = gap_threat
        cap = edge_cap if threat is None else min(edge_cap, self._stop_cap(threat[0], threat[2]))
        return self._decision(transition['state'], path, cap, threat, reason)

    # ------------------------------------------------------------------
    # Highway sectors: compact following + MOBIL, collision stops take priority
    # ------------------------------------------------------------------
    def _relative_speed_limit(self, gap, lead_v, decel, clearance):
        """Speed that can shed closing speed before the bumper clearance.

        A same-speed leader does not require braking merely because its
        absolute speed is high. A slow/stopped leader still requires early
        braking when the closing speed consumes the remaining distance.
        """
        room = max(0., gap-clearance)
        reaction = max(self.reaction_s, self.idm_time_gap_s)
        closing = math.sqrt((decel*reaction)**2+2.*decel*room)-decel*reaction
        return max(0., lead_v)+closing

    def _highway_collision_speed(self, gap, lead_v):
        if gap <= self.margin:
            return 0.
        return self._relative_speed_limit(gap, lead_v, 6., self.margin)

    def _idm(self, v, v0, gap, lead_v):
        """Compact highway following: 3 m gap, closing-speed braking.

        Keep the existing acceleration/incentive API, but do not impose
        the former v*T headway on same-speed traffic.
        """
        v = max(0., v)
        free = self.idm_accel_mps2*(1.-(v/v0)**4 if math.isfinite(v0) and v0 > .1 else 1.)
        if lead_v is None:
            return free
        limit = self._relative_speed_limit(gap, lead_v,
                    self.idm_comfort_decel_mps2, self.idm_min_gap_m)
        requested = (limit-v)/self.idm_command_horizon_s
        return min(free, max(-self.idm_comfort_decel_mps2, requested))

    def _idm_cap(self, v, v0, lead):
        """Comfort following with a cruise floor, bounded by collision speed."""
        if lead is None:
            return float('inf')
        gap = lead[0]-self.front_extent
        a = self._idm(v, v0, gap, lead[2])
        desired = v0 if math.isfinite(v0) else max(v, self.sector_min_speed_kmh/3.6)
        floor = min(self.sector_min_speed_kmh/3.6, desired)
        target = max(floor, v+a*self.idm_command_horizon_s)
        hard = self._highway_collision_speed(gap, lead[2])
        if hard < floor or v > hard+.25:
            self.highway_emergency = True
        return min(target, hard)*3.6

    def _highway_length(self, lateral, speed_mps):
        """Quintic length within the lateral-acceleration and heading limits."""
        lateral = max(.1, lateral)
        by_accel = max(0., speed_mps)*math.sqrt(
            QUINTIC_PEAK_CURVATURE*lateral/self.lane_change_lateral_accel)
        heading = math.radians(min(max(self.lane_change_max_heading_deg, 1.), 45.))
        return max(by_accel, QUINTIC_PEAK_SLOPE*lateral/math.tan(heading))

    def _turn_limited_length(self, lateral):
        """Shortest quintic whose peak curvature the steering still reaches."""
        return math.sqrt(QUINTIC_PEAK_CURVATURE*max(.1, lateral)*self.min_turn_radius_m)

    def _highway_merge_length(self, start, end, speed_mps=0.):
        return self._highway_length(distance(start, end), speed_mps)

    def _length_speed(self, lateral, length):
        """Highest speed at which a quintic of this length keeps the
        lateral-acceleration limit."""
        return length*math.sqrt(self.lane_change_lateral_accel /
                                (QUINTIC_PEAK_CURVATURE*max(.1, lateral)))

    def _lateral_offset(self, index, ego_xy, point):
        tangent = self._tangent(index)
        return (point[1]-ego_xy[1])*tangent[0]-(point[0]-ego_xy[0])*tangent[1]

    def _follower(self, index, ego_xy, centre, obstacles, assumed_mps):
        """Nearest object fully behind the rear bumper in the lane at lateral
        offset centre: (bumper gap, speed along the route). Confirmed tracks
        use their measured estimate; other inputs use assumed_mps."""
        tangent = self._tangent(index)
        normal = (-tangent[1], tangent[0])
        band = self.half_width+self.margin
        best = None
        for obstacle in obstacles:
            longitudinal, lateral = [], []
            for cx, cy in obstacle.corners():
                ox, oy = cx-ego_xy[0], cy-ego_xy[1]
                longitudinal.append(ox*tangent[0]+oy*tangent[1])
                lateral.append(ox*normal[0]+oy*normal[1])
            gap = -self.rear_extent-max(longitudinal)
            if gap <= 0. or min(lateral) > centre+band or max(lateral) < centre-band:
                continue
            # A confirmed track's estimate (even if still settling) is better
            # than a guess; the 2026-10-18 run showed unconfirmed rear clutter
            # assumed at road speed blocking every change.
            speed = (obstacle.vx*tangent[0]+obstacle.vy*tangent[1]
                     if obstacle.confirmed else assumed_mps)
            if best is None or gap < best[0]:
                best = (gap, speed)
        return best

    def _follower_desired(self, speed, v, v0):
        return max(speed, v0 if math.isfinite(v0) else v)

    def _mobil(self, current, lane, v, v0, leaders, followers, through_lane=None):
        """MOBIL incentive and safety of moving from current to lane.

        leaders/followers: lane -> (bumper gap, speed) or None. A lane end is
        a standing leader, so an ending lane is left like a blocked one.
        """
        floor = -2.*self.mobil_safe_decel_mps2

        def accel(speed, desired, lead):
            # Clamped: beyond physical braking a deficit only means unsafe.
            return max(floor, self._idm(speed, desired, lead[0], lead[1]) if lead is not None
                       else self._idm(speed, desired, 0., None))
        body = self.front_extent+self.rear_extent

        def through(follower, lead):
            # The follower's leader when the ego is not between them.
            return None if lead is None else (follower[0]+body+lead[0], lead[1])
        lead_c, lead_l = leaders[current], leaders[lane]
        if through_lane is not None:
            # Only crossing this lane: its leader matters only if reached
            # before the next step into through_lane is done.
            nxt = leaders.get(through_lane)
            lead_l = nxt if nxt is not None and (lead_l is None or nxt[0] < lead_l[0]) else (
                None if lead_l is None or self._passes_through(v, lead_l) else lead_l)
        a_current = accel(v, v0, lead_c)
        a_lane = accel(v, v0, lead_l)
        gain_followers = 0.
        safe = (leaders[lane] is None or
                (leaders[lane][0] >= self.lane_change_gap_m
                 and v <= self._highway_collision_speed(*leaders[lane])))
        new = followers.get(lane)
        if new is not None:
            desired = self._follower_desired(new[1], v, v0)
            after = accel(new[1], desired, (new[0], v))
            gain_followers += after-accel(new[1], desired, through(new, lead_l))
            safe = (safe and new[0] >= self.lane_change_gap_m
                    and new[1] <= self._highway_collision_speed(new[0], v))
        old = followers.get(current)
        if old is not None:
            desired = self._follower_desired(old[1], v, v0)
            gain_followers += (accel(old[1], desired, through(old, lead_c)) -
                               accel(old[1], desired, (old[0], v)))
        pref = self.highway_preferred_lane
        toward = abs(lane-pref) < abs(current-pref)
        bias = (self.highway_lane_bias_mps2 if toward else
                -self.highway_lane_bias_mps2 if abs(lane-pref) > abs(current-pref) else 0.)
        incentive = a_lane-a_current+self.mobil_politeness*gain_followers+bias
        # A slower car remains worth passing before the compact following
        # controller has to brake. Do not lose the existing overtaking
        # behavior merely because the shorter cruise gap is still open.
        if lead_c is not None:
            next_speed = v0 if lead_l is None else min(v0, lead_l[1])
            incentive += min(self.idm_accel_mps2,
                             .2*max(0., next_speed-lead_c[1]-self.slow_lead_margin_mps))
        return incentive, safe, toward

    def _passes_through(self, v, lead):
        """True if a leader in a lane only crossed is not caught up with
        before two one-lane changes are complete."""
        gap, lead_v = lead
        duration = 2.*self._highway_length(3.6, v)/max(v, 1.)
        return gap-max(0., v-lead_v)*duration > self.idm_min_gap_m+self.lane_change_gap_m

    def _highway_change(self, ahead, ego_xy, progress, current, lane, available,
                        obstacles, v, v0):
        """Quintic change, shortened (and slowed) until nothing on the swept
        path needs more than the safe deceleration. No speed precondition."""
        lateral = distance(ego_xy, available[lane])
        minimum = self._highway_length(lateral, 0.)
        planning_speed = max(v, min(v0, self.sector_min_speed_kmh/3.6))
        length = min(self._highway_length(lateral, planning_speed), self._span(current, ahead),
                     self._span(lane, ahead)-8.)
        if length < minimum:
            # The lane ends first: a short, slow merge within the steering limit.
            minimum = min(minimum, self._turn_limited_length(lateral))
        while length >= minimum-1e-6:
            transition = dict(source=current, target=lane, start=progress, length=length,
                              state='', shape='quintic', highway=True, speed_cap_kmh=None)
            path = self._path(ahead, ego_xy, lane, progress, transition)
            speed = self._length_speed(lateral, length)
            if len(path) >= 2:
                lead = self._lead(path, ego_xy, obstacles, self.highway_obstacle_horizon_m)
                if lead is None or min(v, speed) <= self._highway_collision_speed(
                        lead[0]-self.front_extent, lead[2]):
                    # A short merge planned from standstill must keep its
                    # curvature speed limit as the controller accelerates.
                    requested = max(v, v0) if math.isfinite(v0) else v
                    if speed < requested-.05:
                        transition['speed_cap_kmh'] = speed*3.6
                    return transition, path, lead
            if length <= minimum+1e-6:
                break
            length = max(minimum, length*.85)
        return None

    def _target_lane_danger(self, index, ego_xy, v, v0, assumed, available, lane, obstacles):
        blocker = self._gap_blocker(index, ego_xy, available, lane, obstacles)
        if blocker is not None and self._closing(index, ego_xy, v, blocker[0]):
            return blocker
        follower = self._follower(index, ego_xy, self._lateral_offset(index, ego_xy, available[lane]),
                                  obstacles, assumed)
        if follower is not None and follower[1] > self._highway_collision_speed(follower[0], v):
            return None, -follower[0]-self.rear_extent-self.front_extent
        return None

    def _highway_transition(self, index, ego_xy, v, v0, assumed, progress, ahead,
                            available, obstacles, edge_cap, tracked):
        """Continue, finish or abort the active change. None: it is finished
        (or its geometry is gone) and lane selection runs again."""
        transition = self.transition
        lane = transition['target']
        if lane not in available or progress >= transition['start']+transition['length']:
            self.transition = None
            if lane in available:
                self.current_lane = lane
            self.highway_change_end = progress
            return None
        reason = ''
        if transition['state'] != 'CANCELLING':
            danger = self._target_lane_danger(index, ego_xy, v, v0, assumed, available,
                                              lane, tracked)
            # Something revealed on the swept path (e.g. a parked car beyond
            # the sensor range when the change began) that needs more than
            # the safe deceleration: steering back is the way out at speed.
            path = self._path(ahead, ego_xy, lane, progress, transition)
            lead = self._lead(path, ego_xy, tracked, self.highway_obstacle_horizon_m)
            hazard = lead is not None and v > self._highway_collision_speed(
                lead[0]-self.front_extent, lead[2])+.25
            self.cancel_votes = self.cancel_votes+1 if danger is not None or hazard else 0
            fraction = (progress-transition['start'])/max(transition['length'], 1e-6)
            source = transition['source']
            if danger is not None or hazard:
                if danger is not None and danger[0] is not None:
                    self.gap_blocker = danger
                reason = 'lane_change_hazard_ahead' if hazard else 'lane_change_gap_hold'
                back = (self.cancel_votes >= self.cancel_confirm_scans and source in available
                        and (hazard or fraction <= self.cancel_max_progress)
                        and self._gap_blocker(index, ego_xy, available, source, tracked) is None)
                change = (self._highway_change(ahead, ego_xy, progress, lane, source, available,
                                               tracked, v, v0) if back else None)
                if change is not None:
                    # Abort: steer back to the lane we left.
                    self.transition = transition = dict(change[0], state='CANCELLING')
                    self.departed_preferred = source != self.effective_preferred_lane
                    self.cancel_votes = 0
                    lane = source
                    reason = 'lane_change_cancelled'
        path = self._path(ahead, ego_xy, lane, progress, transition)
        if len(path) < 2:
            # Never stop on the highway for missing geometry: drop the change.
            self.transition = None
            return None
        horizon = self.highway_obstacle_horizon_m
        self.active_lane = lane if lane != self.route_lanes[index] else None
        self.obstacle_horizon_path = path_prefix(path, horizon)
        lead = self._lead(path, ego_xy, obstacles, horizon)
        cap = min(edge_cap, self._idm_cap(v, v0, lead))
        if transition.get('speed_cap_kmh') is not None:
            cap = min(cap, transition['speed_cap_kmh'])
        return self._decision(transition['state'], path, cap, lead, reason)

    def _highway_lane_end_cap(self, ahead, available):
        """Lane drop ahead (S11: R1 and R2 end, R3 continues): keep the speed
        from which the remaining one-lane steps still fit at low speed. An
        active change toward the continuing lane is never brought to a halt:
        its length was fitted inside the ending lane when it started."""
        self.lane_end_imminent = False
        current = self.current_lane
        spans = {n: self._span(n, ahead) for n in available}
        span = spans[current]
        longest = max(spans.values())
        if span >= ahead[-1][2]-1e-6 or longest <= span+8.:
            return float('inf')
        # The lane that really continues (R1 ends a metre after R2 in S11).
        target = min((n for n in available if spans[n] >= longest-8.),
                     key=lambda n: abs(n-current))
        step = 1 if target > current else -1
        transition = self.transition
        via = current
        if (transition is not None and transition['target'] in available
                and (transition['target']-transition['source'])*step > 0):
            via = transition['target']
        if via == target:
            return float('inf')
        neighbour = via+step
        spacing = (distance(available[via], available[neighbour])
                   if via in available and neighbour in available else 3.6)
        room = spans.get(via, span)-abs(target-via)*self._turn_limited_length(spacing)-8.
        cap = self._stop_cap(room+self.front_extent+self.margin) if room > 0. else 0.
        # Lets the follower's S11 floor yield to this envelope, so the car
        # can fall in behind traffic in the continuing lane.
        self.lane_end_imminent = cap < float('inf')
        return cap

    def _plan_highway(self, index, ego_xy, speed_mps, obstacles, desired_speed_mps,
                      ahead, progress, available):
        v = max(0., speed_mps)
        self.highway_emergency = False
        self.highway_road_returns_ignored = sum(o.road_surface for o in obstacles)
        obstacles = [o for o in obstacles if not o.road_surface]
        v0 = float('inf') if desired_speed_mps is None else max(desired_speed_mps, .1)
        base = self.route_lanes[index]
        pref = self.highway_preferred_lane
        preferred = min(available, key=lambda n: (abs(n-pref), n != base))
        self.effective_preferred_lane = self.previous_preferred_lane = preferred
        self.preference_reason = 'highway_preferred_lane'
        self.lane_hold = None
        # Lane changes are decided on confirmed tracks only. Flickering
        # unconfirmed returns (rear road clutter, 10 m wide ground patches)
        # cancelled every change in the 2026-10-18 21:29 run. Following the
        # current lane's leader still uses every object.
        tracked = [o for o in obstacles if o.confirmed]
        assumed = v
        edge_cap = self._highway_lane_end_cap(ahead, available)
        if self.transition is not None and edge_cap <= .01:
            # A change that ends standing at a lane end never completes (the
            # run stopped for good cancelling back into ending R1): drop it
            # and pick the merge toward the continuing lane again.
            self.transition = None
            edge_cap = self._highway_lane_end_cap(ahead, available)
        horizon = self.highway_obstacle_horizon_m
        if self.transition is not None:
            decision = self._highway_transition(index, ego_xy, v, v0, assumed, progress, ahead,
                                                available, obstacles, edge_cap, tracked)
            if decision is not None:
                return decision
            edge_cap = self._highway_lane_end_cap(ahead, available)

        current = self.current_lane
        current_path = self._path(ahead, ego_xy, current, progress)
        self.obstacle_horizon_path = path_prefix(current_path, horizon)
        threat = self._lead(current_path, ego_xy, obstacles, horizon,
                            lane=current) if len(current_path) >= 2 else None
        end = ahead[-1][2]
        leaders, followers, spans = {}, {}, {}
        for n in available:
            path = current_path if n == current else self._path(ahead, ego_xy, n, progress)
            lead = threat if n == current else (
                self._lead(path, ego_xy, tracked, horizon, lane=n) if len(path) >= 2 else None)
            item = None if lead is None else (lead[0]-self.front_extent, lead[2])
            spans[n] = self._span(n, ahead)
            if spans[n] < end-1e-6 and (item is None or spans[n]-self.front_extent < item[0]):
                item = (spans[n]-self.front_extent, 0.)
            leaders[n] = item
            followers[n] = self._follower(index, ego_xy,
                                          self._lateral_offset(index, ego_xy, available[n]),
                                          tracked, assumed)
        ending = spans[current] < end-1e-6
        held = (self.highway_change_end is not None and
                0. <= progress-self.highway_change_end < self.highway_change_hold_m)
        best, waiting = None, ''
        for lane in leaders:
            if abs(lane-current) != 1:
                continue
            step = lane-current
            through = (lane+step if abs(lane+step-pref) < abs(lane-pref)
                       and lane+step in leaders else None)
            incentive, safe, toward = self._mobil(current, lane, v, v0, leaders, followers,
                                                  through)
            if (toward and self.departed_preferred and not ending
                    and leaders[lane] is not None
                    and leaders[lane][1] < v0-self.slow_lead_margin_mps):
                # Complete the pass before taking the preferred lane back;
                # a compact safe gap alone is not evidence we passed the car.
                waiting = 'overtake_not_passed'
                continue
            if not toward and not ending and not self._slow(threat, v0):
                # IDM keeps a long gap behind a leader near the desired
                # speed; only a clearly slower one is worth passing.
                continue
            if not safe:
                waiting = 'mobil_unsafe'
                continue
            if incentive <= self.mobil_threshold_mps2:
                continue
            if held and not toward and not ending:
                waiting = waiting or 'highway_change_hold'
                continue
            # Even a new track must keep the immediate bumper/side gap clear.
            blocker = self._gap_blocker(index, ego_xy, available, lane, obstacles)
            if blocker is not None:
                self.gap_blocker = blocker
                waiting = 'lane_change_gap'
                continue
            change = self._highway_change(ahead, ego_xy, progress, current, lane, available,
                                          tracked, v, v0)
            if change is None:
                waiting = waiting or 'lane_change_no_room'
                continue
            if best is None or incentive > best[0]:
                best = (incentive, lane, toward, change)
        if best is not None:
            _, lane, toward, (transition, path, lead) = best
            state = ('RETURNING' if toward and self.departed_preferred
                     else 'LANE_CHANGING' if toward else 'OVERTAKING')
            transition['state'] = state
            self.transition = transition
            self.cancel_votes = 0
            # Departed only by moving away from the preferred lane; R1 -> R2
            # on the way to R3 is a lane change, not an overtake.
            self.departed_preferred = (not toward) or (self.departed_preferred and lane != preferred)
            self.active_lane = lane if lane != base else None
            self.obstacle_horizon_path = path_prefix(path, horizon)
            # The selected merge can leave the ending lane in time. Recompute
            # its envelope now; the old keep-lane cap can be zero at standstill.
            edge_cap = self._highway_lane_end_cap(ahead, available)
            # Track confirmation controls lane selection, not collision braking
            # on the actual path (including this first scan of a new change).
            lead = self._lead(path, ego_xy, obstacles, horizon)
            cap = min(edge_cap, self._idm_cap(v, v0, lead))
            if transition['speed_cap_kmh'] is not None:
                cap = min(cap, transition['speed_cap_kmh'])
            return self._decision(state, path, cap, lead or threat, 'mobil_gain')

        # Keep the lane with compact following; collision and geometry
        # envelopes can still override the ordinary cruise floor.
        self.active_lane = current if current != base else None
        if current == preferred:
            self.departed_preferred = False
        path = (current_path if len(current_path) >= 2 and
                (current != base or distance(ego_xy, available[base]) > .35) else [])
        cap = min(edge_cap, self._idm_cap(v, v0, threat))
        if threat is not None:
            return self._decision('TAILING', path, cap, threat, waiting or 'highway_idm')
        state = 'LANE_WAIT' if waiting else 'OVERTAKING' if self.departed_preferred else 'RACELINE'
        return self._decision(state, path, cap, reason=waiting)
