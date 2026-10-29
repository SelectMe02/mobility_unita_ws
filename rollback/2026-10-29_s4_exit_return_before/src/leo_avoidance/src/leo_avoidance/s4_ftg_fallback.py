"""S4-only short-horizon FTG fallback with validated, latched handoffs.

Never erase obstacle memory or bypass traffic caps. FTG supplies direction;
its usable forward prefix must fit the saved corridor and vehicle body.
"""
import math
import time
from dataclasses import replace
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from leo_avoidance.ftg_local import FollowTheGap, Config
from leo_avoidance.route_avoidance import stopping_speed


class Profile:
    def __init__(self, knots, values, slopes):
        self.start_s, self.end_s = knots[0], knots[-1]
        self.curve = CubicHermiteSpline(knots, values, slopes)

    def offset_at(self, s):
        edge = float(np.clip(s, self.start_s, self.end_s))
        return float(self.curve(edge)+(s-edge)*self.curve(edge, 1))


class S4Fallback:
    def __init__(self, lidar_offset=1.4):
        self.lidar_offset = lidar_offset
        self.ftg = FollowTheGap(Config(lidar_to_rear_x_m=lidar_offset,
            max_speed_mps=25./3.6, max_range_m=30., local_planning_range_m=20.))
        self.reset()

    def reset(self):
        self.mode = 'route'
        self.last_stamp = None
        self.bad_since = self.good_since = None
        self.bad_scans = self.good_scans = 0
        self.entered = None
        self.profile = None
        self.join = None
        self.status = dict(source='LOCAL_PATH', state='IDLE', safe_distance_m=0., reason='')

    def safe(self, planner, profile, items, speed, points, pose):
        end, _ = self.clear_prefix(planner, profile, items, speed, points, pose)
        return profile is not None and end >= profile.end_s-1e-6

    def clear_prefix(self, planner, profile, items, speed, points, pose):
        """Rear-axle progress certified by body, corridor and current scan.

        A distant conflict bounds progress instead of vetoing the whole path.
        Unknown remembered footprints remain obstacles, never presumed empty.
        """
        s = planner.last_s
        if profile is None or profile.end_s <= s:
            return s, 'no forward path'
        low, high = sorted((0., planner.static_pass_zone[4]))
        # A centreline is not a road edge. Permit only a small existing
        # tracking error that monotonically returns into the saved corridor.
        # Keep normal paths and all body/obstacle checks unchanged.
        cfg = planner.config
        allowance = max(0., min(.2, abs(planner.static_pass_zone[4])*.5
                                - cfg.vehicle_half_width_m-cfg.side_margin_m))
        outside = lambda d: max(low-d, d-high, 0.)
        start_d = profile.offset_at(s)
        lo, hi = planner.route.window(s-2., s+3.)
        _, ego_d, _ = planner.route.project(pose[0], pose[1], lo, hi)
        join = min(s+8., profile.end_s)
        errors = [outside(profile.offset_at(a))
                  for a in np.r_[np.arange(s, join, .25), join]]
        returning = (0. < errors[0] <= allowance and abs(start_d-ego_d) <= .05
                     and errors[-1] <= 1e-6
                     and all(b <= a+1e-6 for a, b in zip(errors, errors[1:])))
        end, reason = profile.end_s, 'path end'
        clear, blocker, hit, _ = planner._s4_swept_check(profile, items, s, speed)
        if not clear:
            end = max(s, hit-.25)
            reason = ('remembered' if blocker and blocker.get('remembered') else 'observed') + (
                ' obstacle #'+str(blocker['id']) if blocker else ' curvature limit')
        samples = np.r_[np.arange(s, end, .25), end]
        # Also test current scan points, including untracked small returns.
        xy = np.asarray(points, float).reshape(-1, 3)[:, :2]
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        px = xy[:, 0]+self.lidar_offset
        world = np.column_stack((pose[0]+c*px-sn*xy[:, 1],
                                 pose[1]+sn*px+c*xy[:, 1]))
        cfg = planner.config
        for a in samples:
            d = profile.offset_at(a)
            if (not low-1e-6 <= d <= high+1e-6
                    and not (returning and a <= join and outside(d) <= errors[0]+1e-6)):
                return max(s, a-.25), 'saved corridor boundary'
            center, segment = planner.route.pose(a, d)
            tangent = planner.route.tangents[segment]
            slope = (profile.offset_at(a+.05)-profile.offset_at(a-.05))/.1
            angle = math.atan2(tangent[1], tangent[0])+math.atan(slope)
            dx, dy = world[:, 0]-center[0], world[:, 1]-center[1]
            along = dx*math.cos(angle)+dy*math.sin(angle)
            lateral = -dx*math.sin(angle)+dy*math.cos(angle)
            if np.any((along >= -cfg.rear_extent_m-.15)
                      & (along <= cfg.front_extent_m+.15)
                      & (abs(lateral) <= cfg.vehicle_half_width_m+cfg.side_margin_m+.15)):
                return max(s, a-.25), 'current LiDAR body overlap'
        return end, reason

    def follow_prefix(self, planner, decision, profile, points, pose, speed, source, limit=math.inf):
        # A completed return joins the saved route; it is not a stop line.
        # Validate the continuation with the same body/scan checks.
        if profile is not None and abs(profile.offset_at(profile.end_s)) < .05:
            base = profile
            class Continuation:
                end_s = min(planner.route.s[-1], max(base.end_s, planner.last_s+planner.config.path_length_m))
                def offset_at(self, a):
                    return base.offset_at(a) if a < base.end_s else 0.
            profile = Continuation()
        end, reason = self.clear_prefix(planner, profile, decision.obstacles, speed, points, pose)
        s, cfg = planner.last_s, planner.config
        room = max(0., end-s)
        # Stop short of the first unsafe rear-axle pose. Do not subtract
        # another front overhang: the swept check already includes it.
        cap = min(limit, cfg.pass_speed_kmh/3.6, decision.traffic_speed_cap_mps)
        if reason != 'path end':
            cap = min(cap, stopping_speed(room-.75, cfg.brake_mps2, cfg.reaction_s))
        if profile is None:
            cap = 0.
        if profile is not None and room > .05:
            class Prefix:
                end_s = end
                offset_at = staticmethod(profile.offset_at)
            cap = min(cap, planner._s4_curve_speed(Prefix(), s, speed))
            path = [planner.route.pose(a, profile.offset_at(a))[0]
                    for a in np.r_[np.arange(s, end, .5), end]]
        else:
            path = decision.path  # Retain the plan for inspection while stopped.
        self.status = dict(source=source, state='TRACKING' if cap > .1 else 'STOP',
                           safe_distance_m=round(room, 2), reason=reason)
        return replace(decision, state='PASSING', path=path, speed_cap_mps=cap,
                       offset_now_m=profile.offset_at(s) if profile else decision.offset_now_m,
                       reason='S4 '+source+' safe prefix: '+reason)

    def apply(self, planner, decision, points, pose, speed, stamp, permitted=True):
        if not planner.static_pass_managed:
            self.reset()
            return decision
        fresh = self.last_stamp is None or stamp > self.last_stamp
        if self.last_stamp is not None and (stamp < self.last_stamp or stamp-self.last_stamp > .5):
            self.reset()
        if not fresh:
            return self.hold(decision, 'awaiting fresh scan')
        self.last_stamp = stamp
        previous_source = self.status.get('source', 'LOCAL_PATH')
        self.status = dict(source='LOCAL_PATH', state='TRACKING' if decision.speed_cap_mps > .1 else 'STOP',
                           safe_distance_m=0., reason=decision.reason)
        eligible = (permitted and decision.traffic_speed_cap_mps > .1
                    and not any(o['moving'] for o in decision.obstacles)
                    and len(points) >= 10 and np.isfinite(points).all())
        normal_profile = self.anchored_route(planner, pose) if eligible else None
        local = (self.follow_prefix(planner, decision, normal_profile, points, pose,
                                    speed, 'LOCAL_PATH') if normal_profile is not None else None)
        local_status = dict(self.status)
        checked = local if local is not None else decision
        blocked = (not checked.path or (checked.speed_cap_mps <= .1
                   and checked.state in ('WAIT_PASS', 'PASSING')))
        if blocked:
            if self.bad_since is None:
                self.bad_since = stamp
            self.bad_scans += 1
        else:
            self.bad_since, self.bad_scans = None, 0
        if self.mode == 'route':
            if not (self.bad_scans >= 3 and stamp-self.bad_since >= .4):
                return local if local is not None else decision
            self.mode, self.entered = 'ftg', stamp
        # Traffic, moving objects and invalid scan input are never bypassed.
        if (not permitted or decision.traffic_speed_cap_mps <= .1
                or any(o['moving'] for o in decision.obstacles)
                or len(points) < 10 or not np.isfinite(points).all()):
            self.good_since, self.good_scans = None, 0
            return self.hold(decision, 'traffic/moving object/scan safety hold')
        s = planner.last_s
        lo, hi = planner.route.window(s-2., s+3.)
        _, d, _ = planner.route.project(pose[0], pose[1], lo, hi)
        _, segment = planner.route.pose(s)
        tangent = planner.route.tangents[segment]
        heading = math.atan2(tangent[1], tangent[0])
        error = math.atan2(math.sin(pose[2]-heading), math.cos(pose[2]-heading))
        if abs(error) > math.radians(35.):
            return self.hold(decision, 'heading outside corridor')
        slope = math.tan(error)
        normal = planner.active
        recovered = normal is not None and not blocked and decision.speed_cap_mps > .5
        self.good_scans = self.good_scans+1 if recovered else 0
        self.good_since = (stamp if self.good_since is None else self.good_since) if recovered else None
        if self.mode == 'recover' and s >= self.join and recovered:
            route_slope = (normal.offset_at(s+.05)-normal.offset_at(s-.05))/.1
            if (abs(d-normal.offset_at(s)) <= .25
                    and abs(error-math.atan(route_slope)) <= math.radians(5.)):
                self.reset()
                self.status = dict(source='LOCAL_PATH', state='TRACKING',
                                   safe_distance_m=0., reason=decision.reason)
                return decision
        if (self.mode == 'ftg' and recovered and self.good_scans >= 5
                and stamp-self.good_since >= .6 and stamp-self.entered >= 1.):
            join = min(s+8., normal.end_s-1.)
            if join-s >= 2.:
                knots = [s]+list(np.arange(join, normal.end_s, 1.))+[normal.end_s]
                values = [d]+[normal.offset_at(a) for a in knots[1:]]
                slopes = [slope]+[(normal.offset_at(a+.05)-normal.offset_at(a-.05))/.1 for a in knots[1:]]
                bridge = Profile(knots, values, slopes)
                if self.safe(planner, bridge, decision.obstacles, speed, points, pose):
                    self.profile, self.mode, self.join = bridge, 'recover', join
        if self.mode == 'recover' and not self.safe(planner, self.profile, decision.obstacles, speed, points, pose):
            # A distant new conflict must not discard a usable connection.
            end, _ = self.clear_prefix(planner, self.profile, decision.obstacles, speed, points, pose)
            if end-s < 1.:
                self.mode, self.entered = 'ftg', stamp
                self.good_since, self.good_scans = None, 0
                self.profile = None
        if self.mode == 'recover':
            return self.follow_prefix(planner, decision, self.profile, points, pose, speed, 'RECOVERY')
        ftg = self.ftg.plan(points, speed, -error, wall_follow=False,
                            ego_pose=pose, stamp=stamp, route_trusted=True)
        if ftg.speed_mps <= 0.:
            if local is not None and local.speed_cap_mps > .1:
                self.status = local_status
                return local
            return self.hold(decision, 'FTG has no safe gap')
        # Keep a still-safe trajectory instead of chasing each scan's gap.
        previous_end, _ = self.clear_prefix(planner, self.profile, decision.obstacles, speed, points, pose)
        if self.profile is None or previous_end-s < 8.:
            best, best_end = self.profile, previous_end
            deadline = time.monotonic()+.06
            bearings = [ftg.target_bearing_rad]+list(ftg.candidate_bearings_rad)
            for bearing, length in [(b, length) for b in dict.fromkeys(bearings)
                                    for length in (16., 12., 8.)][:12]:
                if time.monotonic() > deadline:
                    break
                end = min(s+length, planner.static_pass_zone[3])
                if end-s < 8.:
                    continue
                target_d = d+math.tan(error+bearing)*min(8., length*.5)
                target_d = float(np.clip(target_d, min(0., planner.static_pass_zone[4]), max(0., planner.static_pass_zone[4])))
                candidate = Profile([s, end], [d, target_d], [slope, 0.])
                clear_end, _ = self.clear_prefix(planner, candidate, decision.obstacles, speed, points, pose)
                improvement = 0. if best_end-s < 1. else 1.
                if clear_end-s >= 1. and clear_end > best_end+improvement:
                    best, best_end = candidate, clear_end
            self.profile = best
        if self.profile is None:
            if local is not None and local.speed_cap_mps > .1:
                self.status = local_status
                return local
            return self.hold(decision, 'FTG has no usable safe prefix')
        result = self.follow_prefix(planner, decision, self.profile, points, pose, speed, 'FTG', ftg.speed_mps)
        if (previous_source == 'LOCAL_PATH' and local is not None and local.speed_cap_mps > .1
                and self.status['safe_distance_m'] < local_status['safe_distance_m']+2.):
            # Keep the committed local curve unless FTG buys useful progress.
            # Selecting a shorter prefix merely because FTG is armed jitters.
            self.profile = None
            self.status = local_status
            return local
        if result.speed_cap_mps <= .1 and local is not None and local.speed_cap_mps > .1:
            self.status = local_status
            return local
        return result

    def hold(self, decision, reason):
        self.status = dict(source={'route': 'LOCAL_PATH', 'ftg': 'FTG',
                                   'recover': 'RECOVERY'}[self.mode],
                           state='STOP', safe_distance_m=0., reason=reason)
        return replace(decision, speed_cap_mps=0., reason='S4 '+reason)

    def anchored_route(self, planner, pose):
        normal, s = planner.active, planner.last_s
        if normal is None or normal.end_s-s < 2.:
            return None
        lo, hi = planner.route.window(s-2., s+3.)
        _, d, segment = planner.route.project(pose[0], pose[1], lo, hi)
        tangent = planner.route.tangents[segment]
        heading = math.atan2(tangent[1], tangent[0])
        error = math.atan2(math.sin(pose[2]-heading), math.cos(pose[2]-heading))
        if abs(error) > math.radians(35.):
            return None
        slope = (normal.offset_at(s+.05)-normal.offset_at(s-.05))/.1
        if abs(d-normal.offset_at(s)) <= .15 and abs(error-math.atan(slope)) <= math.radians(3.):
            return normal
        join = min(s+8., normal.end_s-1.)
        if join-s < 2.:
            return None
        knots = [s]+list(np.arange(join, normal.end_s, 1.))+[normal.end_s]
        return Profile(knots, [d]+[normal.offset_at(a) for a in knots[1:]],
                       [math.tan(error)]+[(normal.offset_at(a+.05)-normal.offset_at(a-.05))/.1 for a in knots[1:]])
