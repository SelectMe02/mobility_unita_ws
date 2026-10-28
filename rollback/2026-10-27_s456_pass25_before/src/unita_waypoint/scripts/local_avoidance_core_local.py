"""Map-frame obstacle tracking and temporary route planning (no ROS dependency).

The state flow follows the RoboRacer raceline/trailing/overtake/recovery idea.
All geometry and speed limits belong to this vehicle and its saved map.
"""
from dataclasses import dataclass
import math


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smoothstep(t):
    t = max(0., min(1., t))
    return t * t * (3. - 2. * t)


def s_curve(lateral_m, radius_m, max_heading_rad=math.radians(60.)):
    """Two opposite minimum-radius arcs, joined by a straight above max_heading.

    Returns (length_m, arc_long_m, straight_long_m, heading_rad) along the route.
    """
    heading = min(math.acos(max(-1., 1. - lateral_m / (2. * radius_m))), max_heading_rad)
    arc_lateral = radius_m * (1. - math.cos(heading))
    straight = max(0., lateral_m - 2. * arc_lateral) / math.sin(heading) if heading > 0 else 0.
    arc_long = radius_m * math.sin(heading)
    return 2. * arc_long + straight * math.cos(heading), arc_long, straight * math.cos(heading), heading


def s_curve_blend(t, lateral_m, radius_m):
    """Lateral fraction at route fraction t of the s_curve() lane change."""
    t = max(0., min(1., t))
    length, arc_long, straight_long, heading = s_curve(lateral_m, radius_m)
    if lateral_m <= 0. or length <= 0.:
        return 1.

    def first_half(x):
        if x <= arc_long:
            return radius_m - math.sqrt(max(0., radius_m * radius_m - x * x))
        return radius_m * (1. - math.cos(heading)) + (x - arc_long) * math.tan(heading)
    x = t * length
    y = first_half(x) if x <= length / 2. else lateral_m - first_half(length - x)
    return max(0., min(1., y / lateral_m))


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    vx: float = 0.
    vy: float = 0.
    kind: str = 'unknown'


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
                 decel=3.0, lookahead_m=60., pass_speed_kmh=20.,
                 obstacle_horizon_m=45., lane_indices=None, preferred_lane=1,
                 rear_extent=.790, min_rear_gap=8., rear_headway_s=2.,
                 preferred_lane_indices=None, finish_line=None,
                 min_turn_radius_m=3.575, arc_lane_change_max_kmh=15.,
                 lane_change_lateral_accel_mps2=6., r1_tracking_tolerance_m=.5,
                 any_speed_lane_change_ranges=()):
        self.route = list(route)
        self.sectors = list(sectors)
        self.lanes = {int(k): list(v) for k, v in lanes.items()}
        self.lane_indices = lane_indices
        self.half_width = half_width
        self.margin = margin
        self.front_extent = front_extent
        self.reaction_s = reaction_s
        self.decel = decel
        self.lookahead_m = lookahead_m
        self.pass_speed_kmh = pass_speed_kmh
        self.min_turn_radius_m = min_turn_radius_m
        self.arc_lane_change_max_kmh = arc_lane_change_max_kmh
        self.lane_change_lateral_accel_mps2 = lane_change_lateral_accel_mps2
        # Pure Pursuit cuts curves by a few decimetres. Within this offset the
        # car is still in R1 and the corrected R1 path brings it back; beyond
        # it, R1 is re-entered with a lane change (gap and speed checks).
        self.r1_tracking_tolerance_m = r1_tracking_tolerance_m
        self.off_r1 = False
        # Route index intervals where a lane change needs no pass speed.
        self.any_speed_lane_change_ranges = [(int(lo), int(hi))
                                             for lo, hi in any_speed_lane_change_ranges]
        self.obstacle_horizon_m = obstacle_horizon_m
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
        self.min_rear_gap = min_rear_gap
        self.rear_headway_s = rear_headway_s
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
        if min(half_width, margin, front_extent, reaction_s, decel,
               lookahead_m, pass_speed_kmh, obstacle_horizon_m, rear_extent,
               min_rear_gap, rear_headway_s, min_turn_radius_m,
               lane_change_lateral_accel_mps2) <= 0 or arc_lane_change_max_kmh < 0 or self.preferred_lane not in (1, 2, 3):
            raise ValueError('invalid avoidance geometry')
        if not math.isfinite(r1_tracking_tolerance_m) or r1_tracking_tolerance_m < .35:
            raise ValueError('invalid R1 tracking tolerance')
        if any(lo > hi for lo, hi in self.any_speed_lane_change_ranges):
            raise ValueError('invalid any-speed lane change interval')

    def _any_speed_lane_change(self, index):
        return any(lo <= index <= hi for lo, hi in self.any_speed_lane_change_ranges)

    def reset_for_r1_handoff(self):
        """Discard manoeuvres planned while another follower drove the car."""
        self.transition = None
        self.active_lane = None
        self.departed_preferred = False
        self.off_r1 = False
        self.previous_preferred_lane = None

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
            mapping = self.lane_indices.get(number, {})
            if mapping:
                return mapping.get(index)
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
        return point if lane == 1 else self._lane_at(lane, point, index)

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

    def _merge_shape(self, speed_mps):
        return 'arc' if speed_mps <= self.arc_lane_change_max_kmh / 3.6 else 'smooth'

    def _merge_length(self, start, end, speed_mps=0.):
        # Slow: two full-lock arcs (s_curve). Fast: smoothstep, whose peak
        # curvature 6*d/L^2 must stay within the turn radius and whose peak
        # lateral acceleration 6*d*v^2/L^2 within lane_change_lateral_accel_mps2.
        lateral = distance(start, end)
        if self._merge_shape(speed_mps) == 'arc':
            return max(1., s_curve(lateral, self.min_turn_radius_m)[0])
        return max(1., math.sqrt(6. * lateral * self.min_turn_radius_m),
                   speed_mps * math.sqrt(6. * lateral / self.lane_change_lateral_accel_mps2))

    def _blend(self, transition, t):
        if transition.get('shape') == 'arc':
            return s_curve_blend(t, transition['lateral'], self.min_turn_radius_m)
        return smoothstep(t)

    def _r1_offset(self, index, ego_xy):
        """Signed left offset of ego from R1 at this route index."""
        point, tangent = self.route[index], self._tangent(index)
        return (ego_xy[1]-point[1])*tangent[0] - (ego_xy[0]-point[0])*tangent[1]

    def _origin(self, transition, point, index):
        if 'offset' not in transition:
            return self._lane_point(transition['source'], point, index)
        tangent = self._tangent(index)
        return (point[0]-tangent[1]*transition['offset'],
                point[1]+tangent[0]*transition['offset'])

    def _offset_path(self, ahead, ego_xy, offset):
        """Parallel to R1 at a fixed left offset, starting from the actual ego."""
        hold = dict(offset=offset)
        first = self._origin(hold, ahead[0][1], ahead[0][0])
        correction = (ego_xy[0]-first[0], ego_xy[1]-first[1])
        path = []
        for i, point, s in ahead:
            origin = self._origin(hold, point, i)
            decay = 1.-smoothstep(s/8.)
            path.append((origin[0]+correction[0]*decay, origin[1]+correction[1]*decay))
        return path

    def _path(self, ahead, ego_xy, lane, progress, transition=None):
        # The transition has a fixed route anchor. Replanning does not move
        # its end forward, and its first point always matches the actual ego.
        target0 = self._lane_point(lane, ahead[0][1], ahead[0][0])
        if target0 is None:
            return []
        if transition is None:
            correction = (ego_xy[0]-target0[0], ego_xy[1]-target0[1])
        else:
            origin0 = self._origin(transition, ahead[0][1], ahead[0][0])
            if origin0 is None:
                return []
            t = self._blend(transition, (progress-transition['start'])/transition['length'])
            expected = (origin0[0]+t*(target0[0]-origin0[0]),
                        origin0[1]+t*(target0[1]-origin0[1]))
            correction = (ego_xy[0]-expected[0], ego_xy[1]-expected[1])
        path = []
        for i, point, s in ahead:
            target = self._lane_point(lane, point, i)
            if target is None:
                break
            if transition is not None:
                blend = self._blend(transition, (progress+s-transition['start'])/transition['length'])
                origin = self._origin(transition, point, i)
                if origin is None and blend < 1.:
                    break
                if origin is not None:
                    target = (origin[0]+blend*(target[0]-origin[0]),
                              origin[1]+blend*(target[1]-origin[1]))
            decay = 1.-smoothstep(s/8.)
            path.append((target[0]+correction[0]*decay,
                         target[1]+correction[1]*decay))
        return path

    def _lead(self, path, ego_xy, obstacles):
        if len(path) < 2:
            return None
        dx, dy = path[1][0]-path[0][0], path[1][1]-path[0][1]
        norm = max(math.hypot(dx, dy), 1e-6)
        travelled, samples = 0., []
        for a, b in zip(path, path[1:]):
            samples.append((a, b, travelled))
            travelled += distance(a, b)
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
                                    start+t*length, (bx/length, by/length)))
            if not projections:
                continue
            separation, s, tangent = min(projections)
            if s <= self.obstacle_horizon_m and separation < self.half_width+self.margin+obstacle.radius:
                clearance = max(0., s-obstacle.radius)
                velocity = max(0., obstacle.vx*tangent[0]+obstacle.vy*tangent[1])
                found.append((clearance, obstacle, velocity))
        return min(found, key=lambda item: item[0]) if found else None

    def _rear_gap_safe(self, index, ego_xy, target, speed_mps, length, obstacles,
                       ignore_static=False):
        tangent = self._tangent(index)
        normal = (-tangent[1], tangent[0])
        lateral = ((target[0]-ego_xy[0])*normal[0] +
                   (target[1]-ego_xy[1])*normal[1])
        for obstacle in obstacles:
            dx, dy = obstacle.x-ego_xy[0], obstacle.y-ego_xy[1]
            longitudinal = dx*tangent[0]+dy*tangent[1]
            side = dx*normal[0]+dy*normal[1]
            envelope = self.half_width+self.margin+obstacle.radius
            # Include all lanes crossed, including the intermediate lane.
            if not min(0., lateral)-envelope <= side <= max(0., lateral)+envelope:
                continue
            if longitudinal >= 0.:
                continue
            if ignore_static and obstacle.kind == 'static':
                continue
            velocity = obstacle.vx*tangent[0]+obstacle.vy*tangent[1]
            closing = max(0., velocity-speed_mps)
            gap = -longitudinal-self.rear_extent-obstacle.radius-self.margin
            duration = length/max(min(speed_mps, self.pass_speed_kmh/3.6), 1.)
            if (obstacle.kind == 'unknown' or
                    gap < self.min_rear_gap + closing*self.rear_headway_s or
                    (closing > .1 and gap/closing < duration+self.reaction_s)):
                return False, obstacle, max(0., gap)
        return True, None, float('inf')

    def _collision_free(self, path, obstacles, speed_mps, horizon_s=None):
        return math.isinf(self._collision_distance(path, obstacles, speed_mps, horizon_s))

    def _collision_distance(self, path, obstacles, speed_mps, horizon_s=None):
        if len(path) < 2:
            return 0.
        travelled = 0.
        for a, b in zip(path, path[1:]):
            length = distance(a, b)
            if length < 1e-6:
                continue
            dx, dy = (b[0]-a[0])/length, (b[1]-a[1])/length
            # Sample between waypoints, and at <=0.2 seconds at low speed.
            steps = max(1, math.ceil(length/min(.5, max(speed_mps, 1.)*.2)))
            for step in range(steps+1):
                ratio = step/steps
                point = (a[0]+ratio*(b[0]-a[0]), a[1]+ratio*(b[1]-a[1]))
                t = (travelled+ratio*length)/max(speed_mps, 1.)
                if horizon_s is not None and t > horizon_s:
                    return float('inf')
                for obstacle in obstacles:
                    ox = obstacle.x+obstacle.vx*t-point[0]
                    oy = obstacle.y+obstacle.vy*t-point[1]
                    longitudinal, lateral = ox*dx+oy*dy, -ox*dy+oy*dx
                    # Rounded rectangle: asymmetric front/rear vehicle extent.
                    lx = max(-self.rear_extent-longitudinal, 0., longitudinal-self.front_extent)
                    ly = max(0., abs(lateral)-self.half_width)
                    if math.hypot(lx, ly) < obstacle.radius+self.margin:
                        return travelled+ratio*length
            travelled += length
        return float('inf')

    def _stop_cap(self, clearance, lead_speed=0.):
        bumper_gap = clearance-self.front_extent-self.margin
        if bumper_gap <= 0.:
            return 0.
        available = max(0., bumper_gap-lead_speed*1.2)
        safe = math.sqrt((self.decel*self.reaction_s)**2 +
                         2.*self.decel*available)-self.decel*self.reaction_s
        return max(0., (lead_speed+safe)*3.6)

    def _decision(self, state, path, cap, threat=None, reason=''):
        return Decision(state, path, cap, threat[1].kind if threat else '',
                        threat[0] if threat else float('inf'), reason)

    def check_route_pass(self, path, speed_mps, obstacles):
        """Check the selected route pass, without initiating a saved-R1 return.

        Re-evaluate on every scan, including while stopped. An obsolete
        offset correction must not veto a different, collision-free path.
        Keep the old state until the selected path has passed validation.
        """
        collision_distance = self._collision_distance(path, obstacles, speed_mps)
        stopping_distance = speed_mps*self.reaction_s + speed_mps**2/(2.*self.decel)
        if collision_distance <= max(1., stopping_distance):
            return self._decision('BLOCKED', path, 0., reason='route_pass_obstructed')
        if math.isfinite(collision_distance):
            cap = self._stop_cap(collision_distance)
            if cap <= 0.:
                return self._decision('BLOCKED', path, 0., reason='route_pass_obstructed')
            self.reset_for_r1_handoff()
            return self._decision('PASSING', path, min(self.pass_speed_kmh, cap),
                                  reason='route_pass_obstacle_ahead')
        self.reset_for_r1_handoff()
        return self._decision('PASSING', path, self.pass_speed_kmh,
                              reason='selected_route_pass')

    def plan(self, index, ego_xy, speed_mps, obstacles, ego_yaw=None, force_r1=False):
        if force_r1 and self.transition is not None and self.transition['target'] != 1:
            self.reset_for_r1_handoff()
        # See lane endings early enough to decelerate before merging back.
        horizon = max(self.lookahead_m, speed_mps**2/(2.*self.decel)+40.)
        ahead = self._forward(index, horizon)
        if len(ahead) < 2:
            return self._decision('BLOCKED', [], 0., reason='saved_path_ends')
        progress = self._progress_at(index, ego_xy)
        available = {n: self._lane_point(n, ahead[0][1], index)
                     for n in [1]+sorted(self.lanes)}
        available = {n: point for n, point in available.items() if point is not None}
        self.current_lane = min(available, key=lambda n: distance(ego_xy, available[n]))
        requested = self._requested_lane(index, ego_xy, ego_yaw)
        self.preference_reason = 'priority_interval' if requested != self.preferred_lane else 'default_lane'
        if force_r1:
            requested = 1
            self.finish_priority_active = False
            self.preference_reason = 'r1_handoff'
        preferred = requested if requested in available else 1
        if requested not in available:
            self.preference_reason = 'saved_lane_unavailable'
        if self.finish_priority_active:
            # Missing/short geometry must cause slowing or holding, never
            # silently change the user's R3 policy before the finish line.
            preferred = requested
            if preferred not in available:
                self.effective_preferred_lane = preferred
                return self._decision('BLOCKED', [], 0., reason='preferred_saved_path_missing')
            if distance(available[preferred], available[1]) < .35:
                self.current_lane = preferred
        if preferred != 1 and not self.finish_priority_active:
            exit_length = self._merge_length(available[preferred], available[1], speed_mps)
            if self._span(preferred, ahead) < exit_length+8.:
                preferred = 1
                self.preference_reason = 'saved_lane_ending'
        self.effective_preferred_lane = preferred
        if (self.previous_preferred_lane is not None and
                preferred != self.previous_preferred_lane and self.current_lane != preferred):
            self.departed_preferred = True
        self.previous_preferred_lane = preferred
        edge_cap = float('inf')
        if self.current_lane != 1:
            span = self._span(self.current_lane, ahead)
            if span < ahead[-1][2]:
                merge = self._merge_length(available[self.current_lane], available[1], speed_mps)
                edge_cap = math.sqrt((self.pass_speed_kmh/3.6)**2 +
                                     2.*self.decel*max(0., span-merge-8.))*3.6
                if self.finish_priority_active:
                    edge_cap = min(edge_cap, self._stop_cap(span))

        if self.transition is not None:
            lane = self.transition['target']
            point = available.get(lane)
            if (point is not None and progress >= self.transition['start']+self.transition['length']
                    and distance(ego_xy, point) < .5):
                self.transition = None
                self.current_lane = lane
            else:
                path = self._path(ahead, ego_xy, lane, progress, self.transition)
                remaining = max(0., self.transition['start']+self.transition['length']-progress)
                check_time = remaining/max(min(speed_mps, self.pass_speed_kmh/3.6), 1.)
                check_time += self.reaction_s+self.rear_headway_s
                if (not path or distance(path[0], path[-1]) < 8. or
                        not self._collision_free(path, obstacles,
                                                 min(speed_mps, self.pass_speed_kmh/3.6), check_time)):
                    return self._decision('BLOCKED', path, 0., reason='transition_obstructed')
                return self._decision(self.transition['state'], path, self.pass_speed_kmh)

        current = self.current_lane
        # Off every saved lane (e.g. past the end of R2): R1 is the way back.
        r1_offset = self._r1_offset(index, ego_xy)
        # Hysteresis: an offset hovering at the tolerance must not toggle the
        # return decision (and its speed cap) on every plan.
        if current != 1 or abs(r1_offset) <= max(.35, self.r1_tracking_tolerance_m-.3):
            self.off_r1 = False
        elif abs(r1_offset) > self.r1_tracking_tolerance_m:
            self.off_r1 = True
        off_r1 = self.off_r1
        current_path = self._path(ahead, ego_xy, current, progress)
        threat = self._lead(current_path, ego_xy, obstacles)
        # S5 road rings can produce large, jumping single-scan footprints.
        # Keep them in every stopping/gap check, but require the tracker's
        # temporal classification before they can initiate an overtake.
        unconfirmed_s5 = (self.sectors[index] == 5 and threat is not None
                          and threat[1].kind == 'unknown')
        candidates = [preferred]
        if current != preferred and not force_r1:
            candidates.append(current)
        if threat and not force_r1 and not unconfirmed_s5:
            candidates.extend(sorted(available, key=lambda n: distance(available[n], ego_xy)))
        candidates = list(dict.fromkeys(candidates))
        waiting = 'unconfirmed_obstacle' if unconfirmed_s5 else ''
        slow_enough = (speed_mps <= self.pass_speed_kmh/3.6+1.
                       or self._any_speed_lane_change(index))
        for lane in candidates:
            if lane not in available:
                continue
            holding = self._path(ahead, ego_xy, lane, progress)
            if len(holding) < 2 or self._lead(holding, ego_xy, obstacles):
                continue
            changing = lane != current or (lane == 1 and off_r1)
            returning = changing and lane == 1
            transition = None
            if changing:
                length = self._merge_length(ego_xy, available[lane], speed_mps)
                # A return is built from the actual R1 offset, so it never
                # needs the lane it leaves to continue.
                if self._span(lane, ahead) < length+8. or (
                        not returning and self._span(current, ahead) < length):
                    continue
                if not slow_enough:
                    waiting = 'slowing_for_lane_change'
                    continue
                safe, rear, gap = self._rear_gap_safe(index, ego_xy, available[lane],
                                                      speed_mps, length, obstacles,
                                                      ignore_static=returning)
                if not safe:
                    waiting = 'rear_or_crossed_lane_busy'
                    continue
                state = ('RETURNING' if returning or (lane == preferred and self.departed_preferred)
                         else 'OVERTAKING' if threat or (self.departed_preferred and lane != preferred)
                         else 'LANE_CHANGING')
                transition = dict(source=current, target=lane, start=progress,
                                  length=length, state=state,
                                  shape=self._merge_shape(speed_mps),
                                  lateral=distance(ego_xy, available[lane]))
                if returning:
                    transition.update(offset=r1_offset, lateral=abs(r1_offset))
            path = self._path(ahead, ego_xy, lane, progress, transition)
            # During steady driving a faster following vehicle is responsible
            # for its own gap. It must still block every lane-change candidate.
            tangent = self._tangent(index)
            checked = obstacles if changing else [o for o in obstacles if
                (o.x-ego_xy[0])*tangent[0]+(o.y-ego_xy[1])*tangent[1] >= 0.]
            if not self._collision_free(path if changing else path[:2], checked,
                                        min(speed_mps, self.pass_speed_kmh/3.6)
                                        if changing else speed_mps,
                                        length/max(min(speed_mps, self.pass_speed_kmh/3.6), 1.)
                                        +self.reaction_s+self.rear_headway_s if changing else None):
                waiting = 'predicted_path_occupied'
                continue
            self.active_lane = lane if lane != 1 else None
            self.transition = transition
            if changing:
                self.departed_preferred = lane != preferred
                return self._decision(state, path, self.pass_speed_kmh, threat)
            if lane == preferred:
                self.departed_preferred = False
            cap = self.pass_speed_kmh if waiting == 'slowing_for_lane_change' else float('inf')
            cap = min(cap, edge_cap)
            state = ('LANE_WAIT' if waiting else 'OVERTAKING' if self.departed_preferred else 'RACELINE')
            return self._decision(state, path if lane != 1 or distance(ego_xy, available[1]) > .35 else [],
                                  cap, reason=waiting)

        if off_r1:
            # The way back to R1 is blocked: keep the current offset and move on
            # past the blocker, retrying the return on every cycle above.
            self.active_lane = None
            hold = self._offset_path(ahead, ego_xy, r1_offset)
            lead = self._lead(hold, ego_xy, obstacles)
            if lead:
                return self._decision('TAILING', hold, self._stop_cap(lead[0], lead[2]),
                                      lead, 'return_blocked_waiting')
            tangent = self._tangent(index)
            ahead_obstacles = [o for o in obstacles if
                               (o.x-ego_xy[0])*tangent[0]+(o.y-ego_xy[1])*tangent[1] >= 0.]
            if not self._collision_free(hold, ahead_obstacles,
                                        min(max(speed_mps, 1.), self.pass_speed_kmh/3.6),
                                        self.reaction_s+self.rear_headway_s):
                return self._decision('TAILING', hold, 0., reason='return_blocked_waiting')
            return self._decision('LANE_WAIT', hold, self.pass_speed_kmh,
                                  reason='return_blocked_holding_offset')

        # Keep the actual lane while waiting for a merge gap; never snap to R1.
        self.active_lane = current if current != 1 else None
        if threat:
            cap = self._stop_cap(threat[0], threat[2])
            cap = min(cap, edge_cap)
            if not slow_enough and len(available) > 1 and not unconfirmed_s5:
                cap = min(cap, self.pass_speed_kmh)
            return self._decision('TAILING', current_path if current != 1 else [], cap,
                                  threat, waiting)
        return self._decision('BLOCKED', current_path, 0., reason=waiting or 'saved_path_ends')
