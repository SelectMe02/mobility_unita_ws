#!/usr/bin/env python3
"""Offline heading experiments; no ROS node, commands, or simulator connection.

Run after sourcing devel/setup.bash. The bicycle experiment uses perfect
parallel wall returns and constant speed; it tests heading feedback stability,
not tyre/actuator behaviour or closed-loop performance on the recorded drive.
Safety cases compare one step from the SAME baseline planner state so a change
in an earlier wall-following target cannot hide a change in passing behaviour.
"""
import argparse
from collections import Counter
import copy
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'rollback/2026-10-21_s14_heading_before/src/leo_avoidance/src/leo_avoidance/ftg_local.py'
CANDIDATE = ROOT / 'src/leo_avoidance/src/leo_avoidance/ftg_local.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config(module, speed=10.):
    settings = yaml.safe_load((ROOT / 'src/leo_avoidance/config/avoidance_local.yaml').read_text())
    allowed = vars(module.Config())
    settings = {key: value for key, value in settings.items() if key in allowed}
    return module.Config(**dict(settings, max_speed_mps=speed))


def lidar_walls(pose, right=3.846, left=3.846):
    """World-fixed walls, sampled densely; rear axle pose to LiDAR coordinates."""
    x, y, yaw = pose
    along = np.arange(x - 35., x + 36., .15)
    dx = np.tile(along - x, 2)
    dy = np.r_[np.full(len(along), -right - y), np.full(len(along), left - y)]
    c, s = math.cos(yaw), math.sin(yaw)
    points = np.column_stack((c * dx + s * dy - 1.4, -s * dx + c * dy, np.zeros(len(dx))))
    # Isolate wall-follow feedback with the same forward-wall observation
    # envelope as the repository's ideal-wall tests. Rear-wall segmentation
    # artifacts are a separate perception experiment.
    return points[(points[:, 0] >= 1.) & (points[:, 0] <= 24.)]


def steering(bearing):
    # Exact PP target used by DrivingStateNode and gps_waypoint_follower_local.
    x, y = 1.4 + 4. * math.cos(bearing), 4. * math.sin(bearing)
    return math.atan2(2. * 3. * y, x * x + y * y)


def closed_loop(module, speed, y0, yaw0, seconds=20., period=.125):
    clock = [0.]
    planner = module.FollowTheGap(config(module, speed), clock=lambda: clock[0])
    pose = np.array([0., y0, math.radians(yaw0)])
    rows = []
    for i in range(int(seconds / period)):
        clock[0] = i * period
        plan = planner.plan(lidar_walls(pose), speed, -pose[2], wall_follow=True,
                            ego_pose=tuple(pose), stamp=clock[0])
        delta = steering(plan.target_bearing_rad)
        rows.append(dict(t=clock[0], x=float(pose[0]), y=float(pose[1]),
                         yaw_deg=math.degrees(pose[2]), steering_deg=math.degrees(delta),
                         bearing_deg=math.degrees(plan.target_bearing_rad),
                         phase=plan.avoidance_phase, speed_cap=plan.speed_mps,
                         clearance=plan.right_wall_clearance_m, reason=plan.reason))
        # Constant speed isolates lateral feedback; actual speed limits remain recorded.
        # Integrate at 100 Hz between the held scan-rate steering commands.
        steps = max(1, int(math.ceil(period / .01)))
        for _ in range(steps):
            dt = period / steps
            pose[0] += speed * math.cos(pose[2]) * dt
            pose[1] += speed * math.sin(pose[2]) * dt
            pose[2] += speed / 3. * math.tan(delta) * dt
        if abs(pose[1]) > 2.8:
            break
    tail = [r for r in rows if r['t'] >= min(8., seconds / 2.)]
    settled = next((r['t'] for i, r in enumerate(rows)
                    if all(abs(v['y']) <= .1 and abs(v['yaw_deg']) <= 1.
                           for v in rows[i:]) and rows[-1]['t'] - r['t'] >= 1.), None)
    summary = dict(speed_mps=speed, initial_offset_m=y0, initial_heading_deg=yaw0,
                   scan_period_s=period, completed_s=rows[-1]['t'] + period,
                   max_abs_offset_m=max(abs(r['y']) for r in rows),
                   max_abs_heading_deg=max(abs(r['yaw_deg']) for r in rows),
                   tail_offset_rms_m=float(np.sqrt(np.mean([r['y'] ** 2 for r in tail]))) if tail else None,
                   tail_heading_rms_deg=float(np.sqrt(np.mean([r['yaw_deg'] ** 2 for r in tail]))) if tail else None,
                   final_offset_m=rows[-1]['y'], final_heading_deg=rows[-1]['yaw_deg'],
                   settling_time_s=settled,
                   max_steer_step_deg=max(abs(b['steering_deg'] - a['steering_deg'])
                                          for a, b in zip(rows, rows[1:])),
                   phases=dict(Counter(r['phase'] for r in rows)),
                   stop_samples=sum(r['speed_cap'] <= 0 for r in rows))
    return dict(summary=summary, samples=rows)


def safety_sequence():
    walls = lidar_walls((0., 0., 0.))
    def box(x, y, width=1.2):
        return np.array([(x + dx, y + dy, 0.) for dx in (-.2, .2)
                         for dy in np.linspace(-width / 2., width / 2., 9)])
    scenes = []
    for name, points, kwargs in (
            ('wall', walls, {}),
            ('left_pass', np.vstack((walls, box(7.7, -.5))), {}),
            ('right_pass', np.vstack((walls, box(7.7, .8))), {}),
            ('side_contact', np.vstack((walls, [[-.5, 1.1, 0.]])), {}),
            ('blocked_lane', np.vstack((walls, box(2.5, 0., 7.))), {}),
            ('empty_scan', np.zeros((0, 3)), {}),
            ('exit_heading', walls, dict(exit_heading_error_rad=.15, wall_follow_weight=.5,
                                        route_rejoin_remaining_m=20.)),
            ('route', np.vstack((walls, box(8., .5))), dict(wall_follow=False))):
        for i in range(8):
            scenes.append((name, points, kwargs))
    # A measured crossing sequence exercises temporal motion/yield detection.
    for i in range(16):
        scenes.append(('crossing', np.vstack((walls, box(10., 3. - .15 * i))), {}))
    return scenes


def compare_same_state(baseline, candidate):
    clock = [0.]
    before = baseline.FollowTheGap(config(baseline), clock=lambda: clock[0])
    result = []
    previous_yaw = None
    for i, (name, points, extra) in enumerate(safety_sequence()):
        clock[0] = i * .15
        after = candidate.FollowTheGap(config(candidate), clock=lambda: clock[0])
        after.__dict__.update(copy.deepcopy(before.__dict__))
        if hasattr(after, '_previous_bearing_yaw'):
            after._previous_bearing_yaw = previous_yaw
        # New candidate config fields retain their defaults.
        after.config = replace(config(candidate), **vars(before.config))
        yaw = .04 * math.sin(i / 3.)
        kwargs = dict(wall_follow=True, ego_pose=(0., 0., yaw), stamp=clock[0])
        kwargs.update(extra)
        old = before.plan(points, 2., **kwargs)
        new = after.plan(points, 2., **kwargs)
        previous_yaw = yaw
        a, b = asdict(old), asdict(new)
        changed = [key for key in a if a[key] != b.get(key)]
        preserve = old.avoidance_phase != 'WALL_FOLLOW' or name in (
            'side_contact', 'blocked_lane', 'empty_scan', 'exit_heading')
        result.append(dict(case=name, frame=i, phase=old.avoidance_phase,
                           reason=old.reason, changed_fields=changed,
                           preserve_expected=preserve, before_stop=old.obstacle_stop,
                           after_stop=new.obstacle_stop, before_speed=old.speed_mps,
                           after_speed=new.speed_mps, crossing_threats=old.crossing_threats))
    return dict(samples=result, phases=dict(Counter(r['phase'] for r in result)),
                compared=len(result), differences=sum(bool(r['changed_fields']) for r in result),
                protected_cases=sum(r['preserve_expected'] for r in result),
                protected_differences=[r for r in result if r['preserve_expected'] and r['changed_fields']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('heading_comparison.json'))
    parser.add_argument('--seconds', type=float, default=20.)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline, 'heading_baseline'), load(args.candidate, 'heading_candidate')
    report = dict(baseline=str(args.baseline), candidate=str(args.candidate),
                  limitations=['Ideal bicycle with perfect forward walls (LiDAR x=1..24m) and constant speed; no actuator/tire model.',
                               'Fixed-state safety comparison isolates branch behavior; it is not a vehicle rollout.'],
                  cases=[])
    for speed in (5., 6.5, 8., 10.):
        for offset, yaw in ((.6, 0.), (0., 6.)):
            old = closed_loop(baseline, speed, offset, yaw, args.seconds)
            new = closed_loop(candidate, speed, offset, yaw, args.seconds)
            report['cases'].append(dict(baseline=old, candidate=new))
            print(json.dumps(dict(baseline=old['summary'], candidate=new['summary'])), flush=True)
    report['safety'] = compare_same_state(baseline, candidate)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report['safety'].items() if key != 'samples'}))
    print(args.output)


if __name__ == '__main__':
    main()
