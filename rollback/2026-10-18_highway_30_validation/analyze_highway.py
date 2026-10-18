"""Recorded-input highway comparison, with road evidence from recorded XYZ.

No ROS commands are published. Ego follows its recorded trajectory here;
these comparisons cannot establish collision freedom on a new trajectory.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import rosbag
import rospy
from sensor_msgs import point_cloud2

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT/'src/unita_waypoint/scripts'))
import local_avoidance_core as core
from lidar_obstacles import road_surface_evidence
from map_tunner_core import load_project, primary_route, checkpoint_route_indices, saved_lane_indices

spec = importlib.util.spec_from_file_location('highway_before_30', ROOT/
    'rollback/2026-10-18_highway_30_before/src/unita_waypoint/scripts/local_avoidance_core.py')
old = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = old
spec.loader.exec_module(old)
project = load_project(str(ROOT/'src/unita_waypoint/config/map_tunner_project.json'))
points, sectors, route_lanes = primary_route(project, with_lanes=True)
route = [(q['x'], q['y']) for q in points]
indices = saved_lane_indices(project, points, sectors, route_lanes)
cps = checkpoint_route_indices(project, points, sectors)
lanes = {n: [(q['x'], q['y']) for s in project['sectors']
             for q in s['racelines'].get(str(n), [])] for n in (1, 2, 3)}
preferred = {i: 3 for i, sector in enumerate(sectors) if sector == 11 and i <= cps['11']}
mapped = indices.get(3, {})
start = min(i for i in preferred if i in mapped or route_lanes[i] == 3)
finish = dict(sector=11, lane=3, checkpoint='11', start_index=start, end_index=cps['11'],
              y=route[cps['11']][1], joins_primary=route_lanes[cps['11']] == 3 or
              (cps['11'] in mapped and math.dist(mapped[cps['11']], route[cps['11']]) < .5))

def make(module, revised):
    return module.LocalAvoidancePlanner(route, sectors, lanes, lane_indices=indices,
        route_lanes=route_lanes, preferred_lane=1, preferred_lane_indices=preferred,
        finish_line=finish, highway_sectors=(11,), min_turn_radius_m=3./math.tan(math.radians(29.)),
        sector_min_speed_kmh=30. if revised else 40., idm_time_gap_s=.2 if revised else 1.2,
        idm_min_gap_m=3. if revised else 4., idm_comfort_decel_mps2=1.5 if revised else 2.,
        idm_command_horizon_s=.35 if revised else 1.)

def key(o):
    return tuple(round(o[k], 6) for k in ('x', 'y', 'length', 'width', 'yaw'))

reports = []
for name in ('run_20261018_215638', 'run_20261018_220439'):
    logdir = ROOT/'src/unita_waypoint/log'/name
    rows = []
    for line in (logdir/'events.jsonl').open():
        event = json.loads(line)
        if event['topic'] != '/waypoint_debug/control_decision':
            continue
        data = event['data']
        if data.get('waypoint_index') is not None and sectors[data['waypoint_index']] == 11:
            rows.append(data)
    lo, hi = rows[0]['stamp']-.5, rows[-1]['stamp']+.5
    clouds, statuses, scenes = {}, {}, []
    for path in sorted(logdir.glob('sensors_*.bag')):
        with rosbag.Bag(str(path)) as bag:
            for topic, message, stamp in bag.read_messages(topics=[
                    '/waypoint_debug/obstacle_points', '/waypoint_debug/ground_status', '/waypoint_debug/obstacles'],
                    start_time=rospy.Time.from_sec(lo), end_time=rospy.Time.from_sec(hi)):
                if topic.endswith('obstacle_points'):
                    clouds[round(message.header.stamp.to_sec(), 6)] = np.asarray(list(
                        point_cloud2.read_points(message, field_names=('x','y','z'), skip_nans=True)), float).reshape(-1,3)
                else:
                    d = json.loads(message.data)
                    if topic.endswith('ground_status') and d.get('planes'):
                        statuses[round(d['stamp'], 6)] = d
                    elif topic.endswith('/obstacles'):
                        scenes.append(d)
    evidence, id_evidence, examples = {}, {}, []
    for scene in scenes:
        for o in scene['obstacles']:
            measured = round(o.get('measurement_stamp', scene['stamp']), 6)
            if abs(measured-scene['stamp']) < 1e-4:
                xyz, status = clouds.get(measured), statuses.get(measured)
                surface = False
                if xyz is not None and status is not None:
                    dx, dy = xyz[:,0]-o['x'], xyz[:,1]-o['y']
                    c, s = math.cos(o['yaw']), math.sin(o['yaw'])
                    inside = ((np.abs(dx*c+dy*s) <= o['length']/2+.003)
                              & (np.abs(-dx*s+dy*c) <= o['width']/2+.003))
                    z = xyz[inside,2]
                    surface = road_surface_evidence(z, status['ground_reference_z_m'], status['imu_corrected'])
                    if surface and len(examples) < 8:
                        examples.append(dict(stamp=scene['stamp'], x=o['x'], y=o['y'], length=o['length'],
                                             width=o['width'], z_min=float(z.min()), z_max=float(z.max())))
                id_evidence[o['id']] = surface
            evidence[key(o)] = id_evidence.get(o['id'], False)
    before, after = make(old, False), make(core, True)
    comparisons = []
    fields = ('x','y','radius','vx','vy','kind','confirmed','velocity_reliable','length','width','yaw')
    for data in rows:
        i, ego = data['waypoint_index'], (data['x'], data['y'])
        if None in ego:
            continue
        speed = (data.get('current_speed_kmh') or 0.)/3.6
        desired = data.get('speed_limits', {}).get('profile_kmh') or points[i]['speed_kmh']
        old_obs = [old.Obstacle(**{k:o[k] for k in fields}) for o in data.get('obstacles', [])]
        new_obs = [core.Obstacle(**{k:o[k] for k in fields}, road_surface=evidence.get(key(o), False))
                   for o in data.get('obstacles', [])]
        a = before.plan(i, ego, speed, old_obs, desired_speed_mps=desired/3.6)
        b = after.plan(i, ego, speed, new_obs, desired_speed_mps=desired/3.6)
        comparisons.append(dict(stamp=data['stamp'], index=i, x=ego[0], y=ego[1],
            profile_kmh=desired, speed_kmh=speed*3.6,
            recorded_cap_kmh=data.get('speed_limits',{}).get('local_planner_kmh'),
            old_cap_kmh=a.speed_cap_kmh, new_cap_kmh=b.speed_cap_kmh,
            old_state=a.state, new_state=b.state, new_reason=b.reason,
            emergency=after.highway_emergency, lane_end=after.lane_end_imminent,
            ignored_road=after.highway_road_returns_ignored,
            obstacles=[dict(**{k:o[k] for k in fields}, road_surface=evidence.get(key(o),False))
                       for o in data.get('obstacles', [])]))
    report = dict(run=name, samples=len(comparisons),
        old_zero=sum(r['old_cap_kmh'] < .01 for r in comparisons),
        new_zero=sum(r['new_cap_kmh'] < .01 for r in comparisons),
        old_below30=sum(r['old_cap_kmh'] < 30. for r in comparisons),
        new_below30=sum(r['new_cap_kmh'] < 30. for r in comparisons),
        new_ordinary_below30=sum(r['new_cap_kmh'] < 29.99 and not r['emergency'] and not r['lane_end'] for r in comparisons),
        new_emergency=sum(r['emergency'] for r in comparisons),
        road_evidence_samples=sum(r['ignored_road'] > 0 for r in comparisons),
        ground_examples=examples)
    reports.append(report)
    (OUT/(name+'_comparison.json')).write_text(json.dumps(comparisons)+'\n')
    print(json.dumps(report, indent=2), flush=True)
(OUT/'highway_comparison.json').write_text(json.dumps(reports, indent=2)+'\n')
