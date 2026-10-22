"""Replay recorded route-frame obstacles through the old and new RouteAvoidance.

Both planners start together and receive the same inputs: the obstacles
/leo/route_avoidance published (route frame), the recorded driving pose and
Ego speed. Open loop. Prints every update where the two decisions differ.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rosbag
from leo_guard.route_file import load_lane_changes, load_route_points, load_survey

ROOT = Path(__file__).resolve().parents[2]
REL = 'src/leo_avoidance/src/leo_avoidance/route_avoidance.py'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(mod, project, zone_points):
    points = load_route_points(str(project))[:, :2]
    lanes = load_lane_changes(str(ROOT / 'src/leo_avoidance/config/route_lane_changes.json'), points)
    crosswalks = [(s['s'] + 1., s['s'] + 10.) for s in load_survey(
        str(ROOT / 'src/leo_guard/config/intersections_local.json'), points)
        if s.get('controlled') is True]
    planner = mod.RouteAvoidance(points, mod.RouteAvoidanceConfig(), lanes, crosswalks)
    if zone_points and hasattr(planner, 'set_pedestrian_hold_ranges'):
        ss = [planner.route.project(x, y, 0, len(points) - 1)[0] for x, y in zone_points]
        planner.set_pedestrian_hold_ranges([tuple(ss)])
    return planner


def cap_kmh(decision):
    return None if np.isinf(decision.speed_cap_mps) else round(decision.speed_cap_mps * 3.6, 1)


def replay(bag_path, project, zone_points, t_from, t_to, before_path):
    old = build(module(before_path, 'ra_before'), project, None)
    new = build(module(ROOT / REL, 'ra_after'), project, zone_points)
    items = None
    for planner in (old, new):
        planner._to_route = lambda obstacles, s_ego: [dict(o) for o in items]
    pose = speed = None
    rows, holds = [], 0
    with rosbag.Bag(bag_path) as bag:
        t0 = bag.get_start_time()
        for topic, m, ts in bag.read_messages(['/leo/driving_pose', '/competition/ego_speed_kmh',
                                               '/leo/route_avoidance']):
            t = ts.to_sec() - t0
            if topic == '/leo/driving_pose':
                pose = (m.pose.position.x, m.pose.position.y); continue
            if topic.endswith('kmh'):
                speed = m.data / 3.6; continue
            if pose is None or speed is None or not t_from <= t <= t_to:
                continue
            items = json.loads(m.data)['obstacles']
            a, b = old.update(pose, speed, []), new.update(pose, speed, [])
            holds += new.pedestrian_hold is not None
            if (a.state, cap_kmh(a)) != (b.state, cap_kmh(b)):
                rows.append((round(t, 2), round(speed * 3.6, 1), (a.state, cap_kmh(a)),
                             (b.state, cap_kmh(b)), b.reason[:40]))
    return rows, holds


if __name__ == '__main__':
    bag, project, a, b = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    zone = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else None
    before = sys.argv[6] if len(sys.argv) > 6 else str(
        ROOT / 'rollback/2026-10-22_cp4_cp5_pedestrian_hold_before' / REL)
    rows, holds = replay(bag, project, zone, a, b, before)
    for row in rows:
        print(*row)
    print('differing updates: %d, updates with a pedestrian hold: %d' % (len(rows), holds))
