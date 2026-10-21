#!/usr/bin/env python3
"""Independent point-retention replay of the 23:00:42 raw cloud bag.

Uses captured sector labels only to separate test cases; this does not replace
the node's exact route-progress gate. Each comparison starts from identical
fresh planner state. Diagnostic road-plane inliers and low-return groups are
geometric labels, not simulator ground truth or manually identified objects.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import rosbag
from sensor_msgs import point_cloud2
import yaml

from perception_matrix import ROOT, load, point_keys


def assess(module, config, points, active):
    planner = module.FollowTheGap(config, clock=lambda: 1.)
    planner.debug_capture = True
    kwargs = dict(wall_follow=True, ego_pose=(0., 0., 0.), stamp=1.)
    if active:
        kwargs['ground_reference_z_m'] = -1.61
    plan = planner.plan(points, 2., **kwargs)
    snapshot = planner.debug_snapshot or {}
    selected = snapshot.get('points', np.zeros((0, 3)))
    return asdict(plan), np.asarray(selected), snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT /
                        'rollback/2026-10-21_s14_low_boxes_before/src/leo_avoidance/src/leo_avoidance/ftg_local.py')
    parser.add_argument('--candidate', type=Path, default=ROOT /
                        'src/leo_avoidance/src/leo_avoidance/ftg_local.py')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('recorded_filter_comparison.json'))
    args = parser.parse_args()
    directory = ROOT / 'src/unita_waypoint/log/run_20261021_230042'
    params = yaml.safe_load((directory / 'rosparams.yaml').read_text())['leo_driving_state']
    old, new = load(args.baseline, 'filter_replay_baseline'), load(args.candidate, 'filter_replay_candidate')
    configs = [m.Config(**{k: params.get(k, v) for k, v in vars(m.Config()).items()}) for m in (old, new)]
    evidence = json.loads(Path(__file__).with_name('low_point_stage_evidence.json').read_text())
    evidence = {round(r['stamp'], 5): r for r in evidence['rows']}
    records = []
    with rosbag.Bag(str(directory / 'sensors_2026-10-21-23-00-42_0.bag')) as bag:
        for _, msg, _ in bag.read_messages(topics=['/lidar3D']):
            stamp = msg.header.stamp.to_sec()
            diagnostic = evidence.get(round(stamp, 5))
            if diagnostic is None:
                continue
            points = np.asarray(list(point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)), float).reshape(-1, 3)
            a, pa, sa = assess(old, configs[0], points, False)
            b, pb, sb = assess(new, configs[1], points, False)
            active = diagnostic['sector'] == 14
            c, pc, sc = assess(new, configs[1], points, active)
            kept_before, kept_after = point_keys(pa), point_keys(pc)
            plane = diagnostic['plane_diagnostic']
            road = np.zeros((0, 3))
            if plane is not None:
                residual = points[:, 2] - points[:, :2] @ np.asarray(plane[:2]) - plane[2]
                road = points[(np.abs(residual) <= .05) & (points[:, 0] >= 3.)
                              & (points[:, 0] <= 15.) & (np.abs(points[:, 1]) <= 1.3)]
            candidates = []
            for group in diagnostic['low_candidates']:
                lo, hi = np.asarray(group['lidar_min_xyz']), np.asarray(group['lidar_max_xyz'])
                selected = points[np.all(points >= lo - 1e-6, axis=1)
                                  & np.all(points <= hi + 1e-6, axis=1)]
                candidates.append(dict(raw=len(selected),
                    before=sum((kept_before & point_keys(selected)).values()),
                    after=sum((kept_after & point_keys(selected)).values()),
                    center=group['lidar_median_xyz'],
                    height=group['height_above_fitted_road_percentiles']))
            records.append(dict(stamp=stamp, relative_s=diagnostic['relative_s'],
                sector=diagnostic['sector'], active=active,
                recorded_mode=diagnostic['mode'], recorded_phase=diagnostic['ftg_phase'],
                inactive_plan_identical=a == b, inactive_points_identical=np.array_equal(pa, pb),
                baseline_points=len(pa), candidate_points=len(pc),
                baseline_phase=a['avoidance_phase'], candidate_phase=c['avoidance_phase'],
                baseline_reason=a['reason'], candidate_reason=c['reason'],
                baseline_stop=a['obstacle_stop'], candidate_stop=c['obstacle_stop'],
                baseline_road_inliers=sum((kept_before & point_keys(road)).values()),
                candidate_road_inliers=sum((kept_after & point_keys(road)).values()),
                ground_removed_before=sa.get('ground_removed_points'),
                ground_removed_after=sc.get('ground_removed_points'),
                candidates=candidates))
    report = dict(limitations=__doc__,
        baseline=str(args.baseline), candidate=str(args.candidate),
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        bag=str(directory / 'sensors_2026-10-21-23-00-42_0.bag'),
        samples=records, compared=len(records),
        by_sector=dict(Counter(r['sector'] for r in records)),
        inactive_mismatches=[r['stamp'] for r in records
                            if not r['inactive_plan_identical'] or not r['inactive_points_identical']],
        road_inliers_before=sum(r['baseline_road_inliers'] for r in records if r['active']),
        road_inliers_after=sum(r['candidate_road_inliers'] for r in records if r['active']),
        low_points_before=sum(c['before'] for r in records if r['active'] for c in r['candidates']),
        low_points_after=sum(c['after'] for r in records if r['active'] for c in r['candidates']))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'limitations')}))
    print(args.output)


if __name__ == '__main__':
    main()
