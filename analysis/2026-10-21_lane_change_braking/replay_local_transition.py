#!/usr/bin/env python3
"""Read-only replay of recorded transition collision checks; no ROS publishers."""
import json
import math
import sys
from pathlib import Path

import rosbag
import yaml

ROOT = Path('/home/unita/unita_ws')
sys.path.insert(0, str(ROOT / 'src/unita_waypoint/scripts'))
from lane_corridor import LaneCorridor, project_lane_geometry
from local_avoidance_core_local import LocalAvoidancePlanner, Obstacle, distance
from map_tunner_core import primary_route, saved_lane_indices
from obstacle_motion import update_motion


def main():
    project = json.loads((ROOT / 'src/unita_waypoint/log/run_20261021_211600/config/map_tunner_project.json').read_text())
    parameter_path = ROOT / 'src/unita_waypoint/log/run_20261021_211600/rosparams.yaml'
    params = yaml.safe_load(parameter_path.read_text())['leo_waypoint_follower']
    radius = params.get('lane_change_min_turn_radius_m',
                        params['wheelbase'] / math.tan(params['max_steering_rad']))
    route, sectors = primary_route(project, raceline1_only=True)
    xy = [(point['x'], point['y']) for point in route]
    lanes = {number: [(p['x'], p['y']) for sector in project['sectors']
                      for p in sector['racelines'].get(str(number), [])]
             for number in (2, 3)}
    planner = LocalAvoidancePlanner(
        xy, sectors, lanes, lane_indices=saved_lane_indices(project, route, sectors),
        min_turn_radius_m=radius,
        r1_tracking_tolerance_m=params.get('r1_tracking_tolerance_m', .5))
    corridor = LaneCorridor(project_lane_geometry(project, 3.6, True))
    bag_path = ROOT / 'src/unita_waypoint/log/semi_leo_2026-10-21-21-16-00.bag'
    rows, motion_rows, states = [], [], {}
    obstacles, speed, obstacle_received = [], 0., None
    with rosbag.Bag(str(bag_path)) as bag:
        start = bag.get_start_time()
        for topic, message, stamp in bag.read_messages(topics=[
                '/leo/waypoint_debug/obstacles', '/competition/ego_speed_kmh',
                '/leo/waypoint_debug/tracking_state']):
            elapsed = stamp.to_sec() - start
            if topic.endswith('/obstacles'):
                data = json.loads(message.data)
                obstacles = [Obstacle(*(o[k] for k in ('x', 'y', 'radius', 'vx', 'vy')),
                                      kind=o['kind']) for o in data['obstacles']]
                obstacles = [o for o, inside in zip(obstacles,
                    corridor.contains([(o.x, o.y) for o in obstacles])) if inside]
                obstacle_received = elapsed
                for item in data['obstacles']:
                    if item.get('predicted') or 'motion_bounds' not in item:
                        continue
                    state = update_motion(states.get(item['id'], {}), data['stamp'],
                                          item['motion_bounds'])
                    states[item['id']] = state
                    if 86.6 < elapsed < 86.9 or 90.1 < elapsed < 90.65:
                        motion_rows.append(dict(
                            elapsed_s=elapsed, track_id=item['id'],
                            recorded={key: item.get(key) for key in
                                      ('kind', 'vx', 'vy', 'velocity_reliable')},
                            replay={key: state[key] for key in
                                    ('motion_kind', 'vx', 'vy', 'velocity_reliable',
                                     'moving_scans', 'still_scans')}))
            elif topic.endswith('ego_speed_kmh'):
                speed = message.data / 3.6
            elif 84. < elapsed < 96.:
                data = json.loads(message.data)
                if data.get('stop_reason') != 'planner_blocked:transition_obstructed':
                    continue
                index = data['waypoint_index']
                ego = data['x'], data['y']
                transition = data['transition']
                progress = planner._progress_at(index, ego)
                ahead = planner._forward(index, max(planner.lookahead_m,
                                                    speed ** 2 / (2 * planner.decel) + 40))
                path = planner._path(ahead, ego, transition['target'], progress, transition)
                checked_speed = min(speed, planner.pass_speed_kmh / 3.6)
                remaining = max(0., transition['start'] + transition['length'] - progress)
                horizon = remaining / max(checked_speed, 1.) + planner.reaction_s + planner.rear_headway_s
                zero_static = [Obstacle(o.x, o.y, o.radius,
                                        0. if o.kind == 'static' else o.vx,
                                        0. if o.kind == 'static' else o.vy, o.kind)
                               for o in obstacles]
                rows.append(dict(
                    elapsed_s=elapsed, index=index, speed_kmh=speed * 3.6,
                    obstacle_age_s=elapsed - obstacle_received,
                    obstacle_count=len(obstacles),
                    recorded_obstacle_count=data['planner_obstacle_input_count'],
                    path_points=len(path), path_span_m=distance(path[0], path[-1]),
                    horizon_s=horizon, transition=transition,
                    actual_clear=planner._collision_free(path, obstacles, checked_speed, horizon),
                    zero_static_velocity_clear=planner._collision_free(path, zero_static, checked_speed, horizon),
                    obstacles=[vars(o) for o in obstacles]))
    result = dict(
        source_bag=str(bag_path),
        parameter_source=str(parameter_path), min_turn_radius_m=radius,
        method='Rebuild local path from recorded transition and current source; use latest preceding speed and obstacle message, with the same lane-corridor filter.',
        limitation='Recorded subscriber receipt order can differ by milliseconds; obstacle counts agree in all compared samples. Static velocity zero is a diagnostic counterfactual, not a proposed blanket change to moving-obstacle safety.',
        blocked_samples=len(rows),
        same_input_collision_count=sum(not row['actual_clear'] for row in rows),
        zero_static_velocity_clear_count=sum(row['zero_static_velocity_clear'] for row in rows),
        collision_replay=rows, motion_replay=motion_rows)
    output = Path(__file__).with_name('local_transition_collision_replay.json')
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: result[key] for key in
                      ('blocked_samples', 'same_input_collision_count', 'zero_static_velocity_clear_count')}))
    print(output)


if __name__ == '__main__':
    main()
