#!/usr/bin/env python3
"""Replay actual measured detections through the fixed local tracker and planner."""
import json
import math
import sys
from pathlib import Path

import rosbag
import yaml

ROOT = Path('/home/unita/unita_ws')
sys.path.insert(0, str(ROOT / 'src/unita_waypoint/scripts'))
from lane_corridor import LaneCorridor, project_lane_geometry
from lidar_obstacles_local import LidarObstacles
from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle, distance
from map_tunner_core import primary_route, saved_lane_indices

project = json.loads((ROOT / 'src/unita_waypoint/log/run_20261021_211600/config/map_tunner_project.json').read_text())
parameter_path = ROOT / 'src/unita_waypoint/log/run_20261021_211600/rosparams.yaml'
params = yaml.safe_load(parameter_path.read_text())['leo_waypoint_follower']
planner_parameters = dict(
    half_width=params.get('vehicle_half_width_m', .946),
    margin=params.get('avoidance_margin_m', .35),
    front_extent=params.get('front_extent_m', 3.845),
    rear_extent=params.get('rear_extent_m', .790),
    reaction_s=params.get('avoidance_reaction_s', .4),
    decel=params.get('avoidance_decel_mps2', 3.),
    pass_speed_kmh=params.get('pass_speed_kmh', 20.),
    min_turn_radius_m=params.get('lane_change_min_turn_radius_m',
                                params['wheelbase'] / math.tan(params['max_steering_rad'])),
    r1_tracking_tolerance_m=params.get('r1_tracking_tolerance_m', .5),
    arc_lane_change_max_kmh=params.get('arc_lane_change_max_kmh', 15.),
    lane_change_lateral_accel_mps2=params.get('lane_change_lateral_accel_mps2', 6.))
route, sectors = primary_route(project, raceline1_only=True)
xy = [(point['x'], point['y']) for point in route]
lanes = {number: [(p['x'], p['y']) for sector in project['sectors']
                  for p in sector['racelines'].get(str(number), [])]
         for number in (2, 3)}
planner = LocalAvoidancePlanner(
    xy, sectors, lanes, lane_indices=saved_lane_indices(project, route, sectors),
    **planner_parameters)
corridor = LaneCorridor(project_lane_geometry(project, 3.6, True))
detector = LidarObstacles.__new__(LidarObstacles)
detector.tracks, detector.next_id, detector.dynamic_speed_threshold_mps = {}, 1, .8


def obstacles(items):
    result = [Obstacle(*(item[k] for k in ('x', 'y', 'radius', 'vx', 'vy')),
                       kind=item['kind']) for item in items]
    return [o for o, inside in zip(result, corridor.contains([(o.x, o.y) for o in result])) if inside]


bag_path = ROOT / 'src/unita_waypoint/log/semi_leo_2026-10-21-21-16-00.bag'
rows, fixture_frames, fixture_cases = [], [], []
speed, recorded, fixed, fixed_items = 0., [], [], []
with rosbag.Bag(str(bag_path)) as bag:
    start = bag.get_start_time()
    for topic, message, stamp in bag.read_messages(topics=[
            '/leo/waypoint_debug/obstacles', '/competition/ego_speed_kmh',
            '/leo/waypoint_debug/tracking_state']):
        elapsed = stamp.to_sec() - start
        if topic.endswith('/obstacles'):
            data = json.loads(message.data)
            measured = [item for item in data['obstacles'] if not item.get('predicted', False)]
            detections = [[item['x'], item['y'], item['radius'], item['motion_bounds']]
                          for item in measured]
            fixed_items = detector._update_tracks(detections, data['stamp'])
            fixed_measured = [item for item in fixed_items if not item.get('predicted', False)]
            assert len(fixed_measured) == len(measured)
            for original, revised in zip(measured, fixed_measured):
                assert all(original[key] == revised[key] for key in ('x', 'y', 'radius'))
            recorded, fixed = obstacles(data['obstacles']), obstacles(fixed_items)
            if 84. < elapsed < 90.7:
                fixture_frames.append(dict(stamp=data['stamp'], detections=detections))
        elif topic.endswith('ego_speed_kmh'):
            speed = message.data / 3.6
        elif 84. < elapsed < 96.:
            data = json.loads(message.data)
            if data.get('stop_reason') != 'planner_blocked:transition_obstructed':
                continue
            index, ego, transition = data['waypoint_index'], (data['x'], data['y']), data['transition']
            progress = planner._progress_at(index, ego)
            ahead = planner._forward(index, max(planner.lookahead_m, speed ** 2 / (2 * planner.decel) + 40))
            path = planner._path(ahead, ego, transition['target'], progress, transition)
            checked_speed = min(speed, planner.pass_speed_kmh / 3.6)
            remaining = max(0., transition['start'] + transition['length'] - progress)
            horizon = remaining / max(checked_speed, 1.) + planner.reaction_s + planner.rear_headway_s
            original_clear = planner._collision_free(path, recorded, checked_speed, horizon)
            fixed_clear = planner._collision_free(path, fixed, checked_speed, horizon)
            rows.append(dict(elapsed_s=elapsed, original_clear=original_clear, fixed_clear=fixed_clear,
                             recorded=[vars(o) for o in recorded], fixed=fixed_items))
            fixture_cases.append(dict(frame_index=len(fixture_frames)-1, path=path,
                                      speed_mps=checked_speed, horizon_s=horizon,
                                      original_obstacles=[vars(o) for o in recorded]))

report = dict(source_bag=str(bag_path),
              parameter_source=str(parameter_path), planner_parameters=planner_parameters,
              method='Replay every recorded measured center/radius/bounds through the revised LocalLidarObstacles._update_tracks; rebuild each recorded transition path and run the unrelaxed collision checker.',
              geometry='All measured centers and radii are asserted unchanged.',
              samples=len(rows), original_blocked=sum(not r['original_clear'] for r in rows),
              fixed_blocked=sum(not r['fixed_clear'] for r in rows), cases=rows)
result_path = Path(__file__).with_name('motion_fix_collision_replay.json')
result_path.write_text(json.dumps(report, indent=2) + '\n')
fixture = dict(source_bag=bag_path.name,
               planner_parameters=planner_parameters,
               parameter_source='run_20261021_211600/rosparams.yaml: leo_waypoint_follower',
               description='Observed local LiDAR boxes from 84.0 to 90.7 seconds and all nine blocked transition collision paths; no geometry or margins relaxed.',
               frames=fixture_frames, cases=fixture_cases)
fixture_path = ROOT / 'src/unita_waypoint/test/data/static_lane_change_20261021.json'
fixture_path.write_text(json.dumps(fixture, separators=(',', ':')) + '\n')
print(json.dumps({k: report[k] for k in ('samples', 'original_blocked', 'fixed_blocked')}))
print(result_path)
print(fixture_path)
