#!/usr/bin/env python3
"""Compare FTG candidates on recorded tunnel clouds without publishing ROS data.

The bags do not contain the exact odometry/worker inputs. This is a controlled
comparison on reconstructed inputs, NOT an exact execution replay. Every
candidate receives the same cloud, reconstructed guidance, clock, and copied
baseline state; non-wall-follow branches should remain identical.
"""
import argparse
from collections import Counter
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np
import rosbag
from sensor_msgs import point_cloud2
import yaml

from leo_avoidance.route_avoidance import RouteFrame
from compare_heading import ROOT, BASELINE, CANDIDATE, load


def collect(run):
    log = ROOT / 'src/unita_waypoint/log'
    bag_time = '-'.join((run[:2], run[2:4], run[4:]))
    main = log / ('semi_leo_2026-10-21-' + bag_time + '.bag')
    directory = log / ('run_20261021_' + run)
    clouds, states, speeds = [], [], []
    for path in sorted(directory.glob('sensors_*.bag')):
        with rosbag.Bag(str(path)) as bag:
            for _, msg, time in bag.read_messages(topics=['/lidar3D']):
                clouds.append((time.to_sec(), msg.header.stamp.to_sec(), msg))
    clouds.sort(key=lambda item: item[0])
    latest = None
    with rosbag.Bag(str(main)) as bag:
        for topic, msg, time in bag.read_messages(topics=[
                '/leo/driving_status', '/leo/avoidance_status', '/competition/ego_speed_kmh']):
            t = time.to_sec()
            if topic == '/competition/ego_speed_kmh':
                speeds.append((t, msg.data / 3.6))
            elif topic == '/leo/avoidance_status':
                latest = (t, json.loads(msg.data))
            elif (latest is not None and abs(t - latest[0]) < .03
                  and latest[1]['valid']):
                d = json.loads(msg.data)
                if d['mode'] == 'BLACKOUT' and d.get('scan_age_s') is not None:
                    states.append((t, d, latest[1]))
    arrival = np.array([c[0] for c in clouds])
    speed_times = np.array([s[0] for s in speeds])
    selected = {}
    for time, state, plan in states:
        received = time - state['scan_age_s']
        index = int(np.argmin(abs(arrival - received)))
        error = float(abs(arrival[index] - received))
        if error > .04 or index in selected:
            continue
        pose = state.get('lidar_pose_before_select') or state.get('map_pose')
        if pose is None or state.get('scan_fault_reason'):
            continue
        cloud_time, stamp, msg = clouds[index]
        speed = speeds[int(np.argmin(abs(speed_times - cloud_time)))][1]
        selected[index] = dict(stamp=stamp, arrival=cloud_time, pose=pose, speed=speed,
                               points=np.asarray(list(point_cloud2.read_points(
                                   msg, field_names=('x', 'y', 'z'), skip_nans=True)), float).reshape(-1, 3),
                               match_error_s=error, recorded_plan=plan)
    project = json.loads((directory / 'config/map_tunner_project.json').read_text())
    from map_tunner_core import primary_route
    points, _ = primary_route(project, raceline1_only=True)
    route = RouteFrame([(p['x'], p['y']) for p in points])
    params = yaml.safe_load((directory / 'rosparams.yaml').read_text())['leo_driving_state']
    return [selected[i] for i in sorted(selected)], route, points, params


def guidance(frame, route, last_s):
    x, y, yaw = frame['pose']
    lo, hi = (0, len(route.lengths)) if last_s is None else route.window(last_s - 15., last_s + 40.)
    s, _, seg = route.project(x, y, lo, hi)
    target, _ = route.pose(s + 5.)
    target_yaw = math.atan2(target[1] - y, target[0] - x)
    preferred = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    live = frame['recorded_plan']
    remaining = live.get('route_rejoin_remaining_m')
    exit_yaw = None
    weight = 1.
    if remaining is not None:
        start, _ = route.pose(s)
        end, _ = route.pose(s + 5.)
        heading = math.atan2(end[1] - start[1], end[0] - start[0])
        exit_yaw = math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))
        # Reconstruct input blend; PASSING's output always reports zero weight.
        blend = float(np.clip((35. - remaining) / (35. - 12.), 0., 1.))
        weight = 1. - blend * blend * (3. - 2. * blend)
    return s, seg, dict(wall_follow=True, wall_follow_weight=weight,
                        route_rejoin_remaining_m=remaining,
                        exit_heading_error_rad=exit_yaw,
                        preferred_bearing_rad=preferred, ego_pose=tuple(frame['pose']),
                        stamp=frame['stamp'])


def compare(run, baseline, candidate):
    frames, route, route_points, params = collect(run)
    old_config = baseline.Config(**{k: params.get(k, v) for k, v in vars(baseline.Config()).items()})
    new_config = candidate.Config(**{k: params.get(k, v) for k, v in vars(candidate.Config()).items()})
    clock = [0.]
    before = baseline.FollowTheGap(old_config, clock=lambda: clock[0])
    records = []
    last_s = previous_yaw = None
    for i, frame in enumerate(frames):
        clock[0] = frame['stamp'] - frames[0]['stamp']
        last_s, index, kwargs = guidance(frame, route, last_s)
        speed_limit = float(route_points[index]['speed_kmh']) / 3.6
        from dataclasses import replace
        before.config = replace(before.config, max_speed_mps=speed_limit)
        after = candidate.FollowTheGap(new_config, clock=lambda: clock[0])
        after.__dict__.update(copy.deepcopy(before.__dict__))
        if hasattr(after, '_previous_bearing_yaw'):
            after._previous_bearing_yaw = previous_yaw
        after.config = replace(new_config, max_speed_mps=speed_limit)
        old = before.plan(frame['points'], frame['speed'], **kwargs)
        new = after.plan(frame['points'], frame['speed'], **kwargs)
        previous_yaw = frame['pose'][2]
        a, b = asdict(old), asdict(new)
        changed = [key for key in a if a[key] != b.get(key)]
        protected = old.avoidance_phase != 'WALL_FOLLOW' or kwargs['exit_heading_error_rad'] is not None
        records.append(dict(index=i, stamp=frame['stamp'], t=clock[0], speed_mps=frame['speed'],
                            match_error_s=frame['match_error_s'], changed_fields=changed,
                            preserve_expected=protected, live_phase=frame['recorded_plan']['avoidance_phase'],
                            baseline=a, candidate=b))
    report = dict(run=run, samples=records, compared=len(records),
                  phases=dict(Counter(r['baseline']['avoidance_phase'] for r in records)),
                  original_phase_agreement=sum(r['baseline']['avoidance_phase'] == r['live_phase'] for r in records),
                  protected_cases=sum(r['preserve_expected'] for r in records),
                  protected_differences=[r['index'] for r in records if r['preserve_expected'] and r['changed_fields']],
                  stop_flag_differences=[r['index'] for r in records
                                         if r['baseline']['obstacle_stop'] != r['candidate']['obstacle_stop']],
                  match_error_p95_s=float(np.percentile([r['match_error_s'] for r in records], 95)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', default=['224142', '224300'])
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('recorded_cloud_comparison.json'))
    args = parser.parse_args()
    baseline, candidate = load(args.baseline, 'bag_heading_baseline'), load(args.candidate, 'bag_heading_candidate')
    # Import the saved project reader without launching any node.
    import sys
    sys.path.insert(0, str(ROOT / 'src/unita_waypoint/scripts'))
    reports = [compare(run, baseline, candidate) for run in args.runs]
    output = dict(limitations=[
        'Raw LiDAR is original. Pose and selected cloud are reconstructed from published driving status.',
        'Exact odometry and preferred-bearing/worker inputs are absent from these bags.',
        'Capture yaw is approximated with recorded map-pose yaw; original phase agreement is reported.',
        'Fixed-state comparisons test branch preservation, not closed-loop driving performance.'], runs=reports)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    for report in reports:
        print(json.dumps({key: value for key, value in report.items() if key != 'samples'}))
    print(args.output)


if __name__ == '__main__':
    main()
