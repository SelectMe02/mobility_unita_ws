#!/usr/bin/env python3
"""Independent low-object/road-retention matrix; never starts ROS nodes.

Pass the new S14-only plan arguments as JSON with --active-kwargs. With that
gate absent, ALL original Plan fields and the actual selected points must be
identical. Active results are measurements, not a claim that every raised
surface is an obstacle; height/support acceptance is decided from bag evidence.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config(module):
    settings = yaml.safe_load((ROOT / 'src/leo_avoidance/config/avoidance_local.yaml').read_text())
    return module.Config(**{key: 10. if key == 'max_speed_mps' else settings.get(key, value)
                            for key, value in vars(module.Config()).items()})


def cloud(height=None, slope_x=0., slope_y=0., noise=0., sparse=False, box_x=8., ground_z=-1.23):
    grid_x, grid_y = np.meshgrid(np.arange(2.5, 14., .25), np.arange(-4., 4.01, .25))
    x, y = grid_x.ravel(), grid_y.ravel()
    rng = np.random.RandomState(14)
    z = ground_z + slope_x * x + slope_y * y + rng.uniform(-noise, noise, len(x))
    road = np.column_stack((x, y, z))[::100 if sparse else 1]
    wx = np.arange(1., 24., .15)
    walls = np.vstack([np.column_stack((wx, np.full(len(wx), y), np.zeros(len(wx))))
                       for y in (-3.846, 3.846)])
    box = np.zeros((0, 3))
    if height is not None:
        bx, by = np.meshgrid(np.linspace(box_x - .4, box_x + .4, 5), np.linspace(-.6, .6, 7))
        bx, by = bx.ravel(), by.ravel()
        box = np.column_stack((bx, by, ground_z + slope_x * bx + slope_y * by + height))
    return np.vstack((road, walls, box)), dict(road=road, box=box)


def point_keys(points):
    return Counter(map(tuple, np.rint(np.asarray(points).reshape(-1, 3) * 1e6).astype(np.int64)))


def run(module, points, labels, kwargs):
    planner = module.FollowTheGap(config(module), clock=lambda: 1.)
    planner.debug_capture = True
    plan = planner.plan(points, 2., wall_follow=True, ego_pose=(0., 0., 0.), stamp=1., **kwargs)
    snapshot = planner.debug_snapshot or {}
    selected = snapshot.get('points', np.zeros((0, 3)))
    kept = point_keys(selected)
    counts = {key + '_retained': sum((kept & point_keys(group)).values())
              for key, group in labels.items()}
    counts.update({key + '_input': len(group) for key, group in labels.items()})
    counts.update(ground_removed=snapshot.get('ground_removed_points'),
                  height_removed=snapshot.get('height_removed_points'),
                  phase=plan.avoidance_phase, reason=plan.reason,
                  front_points=plan.front_points, obstacle_stop=plan.obstacle_stop,
                  speed_mps=plan.speed_mps)
    return asdict(plan), np.asarray(selected), counts


def cases():
    for slope_x, slope_y, noise, sparse in (
            (0., 0., 0., False), (.04, 0., 0., False), (-.04, 0., 0., False),
            (.08, .02, 0., False), (-.08, -.02, 0., False),
            (0., .06, 0., False), (0., 0., .025, False), (0., 0., 0., True)):
        for height in (None, .04, .08, .12, .18, .30):
            yield dict(height=height, slope_x=slope_x, slope_y=slope_y,
                       noise=noise, sparse=sparse, box_x=8.)
    for x in (3., 15., 20.):
        yield dict(height=.12, box_x=x, slope_x=-.04)
    # The latest tunnel scan's measured road height differs from the old
    # nominal mount height. Include that calibration without moving the box.
    for slope_x, noise in ((0., 0.), (0., .025), (.04, 0.), (-.04, 0.)):
        for height in (None, .04, .08, .12, .18, .30):
            yield dict(height=height, ground_z=-1.61, slope_x=slope_x, noise=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Snapshot including the previous heading fix, before low-box changes.')
    parser.add_argument('--candidate', type=Path,
                        default=ROOT / 'src/leo_avoidance/src/leo_avoidance/ftg_local.py')
    parser.add_argument('--active-kwargs', default='{}', help='JSON of S14-only plan arguments.')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('perception_matrix.json'))
    args = parser.parse_args()
    active = json.loads(args.active_kwargs)
    before, after = load(args.baseline, 'low_box_baseline'), load(args.candidate, 'low_box_candidate')
    results = []
    for index, params in enumerate(cases()):
        points, labels = cloud(**params)
        old_plan, old_points, old_counts = run(before, points, labels, {})
        off_plan, off_points, _ = run(after, points, labels, {})
        new_plan, new_points, new_counts = run(after, points, labels, active)
        results.append(dict(index=index, parameters=params, baseline=old_counts, candidate=new_counts,
                            inactive_plan_identical=old_plan == off_plan,
                            inactive_points_identical=np.array_equal(old_points, off_points),
                            active_changed_fields=[key for key in old_plan if old_plan[key] != new_plan[key]]))
    report = dict(baseline=str(args.baseline), candidate=str(args.candidate), active_kwargs=active,
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  inactive_mismatches=[row['index'] for row in results
                                       if not row['inactive_plan_identical'] or not row['inactive_points_identical']],
                  cases=results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cases=len(results), inactive_mismatches=report['inactive_mismatches'])))
    for row in results:
        if row['parameters'].get('height') is None:
            print(json.dumps(row))
    print(args.output)


if __name__ == '__main__':
    main()
