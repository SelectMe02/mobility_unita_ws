"""Open-loop replay of recorded S11 inputs through the old and new planner.

Both planners receive the recorded ego pose/speed/index and the obstacles in
/waypoint_debug/control_decision. Prints the updates whose highway decision
(state, reason, transition target) differs. No ROS output.
"""
import glob
import importlib.util
import json
import math
from pathlib import Path
import sys

import rosbag

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/unita_waypoint/scripts'))
from map_tunner_core import (load_project, primary_route, checkpoint_route_indices,  # noqa: E402
                             saved_lane_indices)
REL = 'src/unita_waypoint/scripts/local_avoidance_core.py'
FIELDS = ('x', 'y', 'radius', 'vx', 'vy', 'kind', 'confirmed', 'velocity_reliable',
          'length', 'width', 'yaw', 'road_surface', 'raceline', 'lane_sector',
          'lane_stable', 'lane_boundary_crossing', 'raw_width', 'predicted')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def build(core, project_file):
    project = load_project(str(project_file))
    points, sectors, route_lanes = primary_route(project, with_lanes=True)
    route = [(float(q['x']), float(q['y'])) for q in points]
    checkpoints = checkpoint_route_indices(project, points, sectors)
    indices = saved_lane_indices(project, points, sectors, route_lanes)
    lanes = {n: [(float(q['x']), float(q['y'])) for s in project['sectors']
                 for q in s['racelines'].get(str(n), [])] for n in (1, 2, 3)}
    end = checkpoints['11']
    preferred = {i: 3 for i, s in enumerate(sectors) if s == 11 and i <= end}
    mapped = indices.get(3, {})
    start = min(i for i in preferred if i in mapped or route_lanes[i] == 3)
    finish = dict(sector=11, lane=3, checkpoint='11', start_index=start, end_index=end,
                  y=route[end][1], joins_primary=route_lanes[end] == 3 or
                  (end in mapped and math.dist(mapped[end], route[end]) < .5))
    planner = core.LocalAvoidancePlanner(route, sectors, lanes, lane_indices=indices,
                                         route_lanes=route_lanes, preferred_lane=1,
                                         preferred_lane_indices=preferred, finish_line=finish,
                                         highway_sectors=(11,))
    return planner, route, sectors, [float(q['speed_kmh']) for q in points]


def frames(run_dir, leo_bag):
    t0 = rosbag.Bag(str(leo_bag)).get_start_time()
    for path in sorted(glob.glob(str(run_dir / 'sensors_*.bag'))):
        for _, m, ts in rosbag.Bag(path).read_messages(['/waypoint_debug/control_decision']):
            yield ts.to_sec() - t0, json.loads(m.data)


def replay(run_dir, leo_bag, before):
    old_core, new_core = module(before, 'lac_before'), module(ROOT / REL, 'lac_after')
    project = run_dir / 'config/map_tunner_project.json'
    (old, route, sectors, speeds), (new, *_) = build(old_core, project), build(new_core, project)
    rows, s11 = [], 0
    for t, d in frames(run_dir, leo_bag):
        index = d.get('waypoint_index')
        if index is None or sectors[index] != 11 or d.get('x') is None:
            continue
        s11 += 1
        a, b = route[index], route[min(index+1, len(route)-1)]
        yaw = math.atan2(b[1]-a[1], b[0]-a[0])
        out = []
        for planner, core in ((old, old_core), (new, new_core)):
            obstacles = [core.Obstacle(**{k: o[k] for k in FIELDS if k in o}) for o in d['obstacles']]
            dec = planner.plan(index, (d['x'], d['y']), (d['current_speed_kmh'] or 0.)/3.6,
                               obstacles, ego_yaw=yaw, desired_speed_mps=speeds[index]/3.6)
            tr = planner.transition
            out.append((dec.state, dec.reason, tr and tr['target']))
        if out[0] != out[1]:
            rows.append((round(t, 2), round(d['current_speed_kmh'] or 0.), out[0], out[1]))
    return rows, s11


if __name__ == '__main__':
    run_dir, leo_bag = Path(sys.argv[1]), Path(sys.argv[2])
    before = sys.argv[3] if len(sys.argv) > 3 else str(
        ROOT / 'rollback/2026-10-22_highway_lane_change_side_before' / REL)
    rows, s11 = replay(run_dir, leo_bag, before)
    for row in rows:
        print(*row)
    print('S11 updates: %d, differing: %d' % (s11, len(rows)))
