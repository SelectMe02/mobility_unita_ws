"""Read recorded scenes; compare box coasting and per-path leader selection.

No commands are published. The ego and traffic keep their recorded poses.
This isolates perception/leader changes, not closed-loop driving performance.
"""
from bisect import bisect_right
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/unita_waypoint/scripts'))
from lane_corridor import LaneCorridor
from lidar_obstacles import LidarObstacles
from local_avoidance_core import LocalAvoidancePlanner, Obstacle
from map_tunner_core import load_project, primary_route, saved_lane_indices

spec = importlib.util.spec_from_file_location('lidar_before_lane_labels', ROOT/
    'rollback/2026-10-18_highway_lane_labels_before/src/unita_waypoint/scripts/lidar_obstacles.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def tracker(cls, corridor):
    n = cls.__new__(cls)
    n.tracks, n.next_id, n.dynamic_speed_threshold_mps = {}, 1, .8
    n.track_confirm_time_s = .2
    n.lane_corridor = corridor
    return n


def obstacle(item):
    fields = Obstacle.__dataclass_fields__
    values = {k: v for k, v in item.items() if k in fields}
    values['track_id'] = item['id']
    return Obstacle(**values)


def analyse(name):
    directory = ROOT/'src/unita_waypoint/log'/name
    events = [json.loads(line) for line in (directory/'events.jsonl').open()]
    geometry = next(e['data'] for e in events if e['topic'] == '/waypoint_debug/lane_geometry')
    corridor = LaneCorridor(geometry)
    before, after = tracker(old.LidarObstacles, corridor), tracker(LidarObstacles, corridor)
    scenes, coasting, labels, measured_growth = [], [], Counter(), []
    previous = {}
    for event in events:
        if event['topic'] != '/waypoint_debug/obstacles':
            continue
        data = event['data']
        stamp = data['stamp']
        measured = [o for o in data['obstacles']
                    if abs(o.get('measurement_stamp', stamp)-stamp) < 1e-4]
        detections = [tuple(o[k] for k in ('x', 'y', 'radius', 'length', 'width', 'yaw'))
                      + (o.get('road_surface', False),) for o in measured]
        a, b = before._update_tracks(detections, stamp), after._update_tracks(detections, stamp)
        for o in measured:
            if o['id'] in previous and abs(o['width']-previous[o['id']]['width']) > .5:
                measured_growth.append(dict(stamp=stamp, id=o['id'], width=o['width'],
                                            previous_width=previous[o['id']]['width']))
            previous[o['id']] = o
        for o in b:
            if o.get('lane_sector') == 11:
                labels[(o['raceline'], o['lane_stable'], o['lane_boundary_crossing'])] += 1
            if o.get('predicted') and o['raw_width']-o['width'] > 1e-6:
                coasting.append(dict(stamp=stamp, id=o['id'], raceline=o['raceline'],
                                     old_width=o['raw_width'], new_width=o['width']))
        scenes.append((stamp, a, b))
    project = load_project(str(directory/'config/map_tunner_project.json'))
    points, sectors, route_lanes = primary_route(project, with_lanes=True)
    indices = saved_lane_indices(project, points, sectors, route_lanes)
    lanes = {n: [(q['x'], q['y']) for s in project['sectors']
                 for q in s['racelines'].get(str(n), [])] for n in (1, 2, 3)}
    planner = LocalAvoidancePlanner([(q['x'], q['y']) for q in points], sectors, lanes,
        route_lanes=route_lanes, lane_indices=indices, highway_sectors=(11,))
    planner.highway_active = True
    stamps = [s[0] for s in scenes]
    selected, changes, brakes = set(), [], Counter()
    evaluated = stable_ignored = 0
    for event in events:
        if event['topic'] != '/waypoint_debug/control_decision':
            continue
        d = event['data']
        index = d.get('waypoint_index')
        if index is None or sectors[index] != 11 or d.get('x') is None or d.get('y') is None:
            continue
        if d['command'].get('brake', 0.) > .01:
            brakes[(d['reason'], d['state'])] += 1
        scene_index = bisect_right(stamps, d['stamp'])-1
        if scene_index < 0 or scene_index in selected or d['stamp']-stamps[scene_index] > .35:
            continue
        selected.add(scene_index)
        ego = (d['x'], d['y'])
        ahead = planner._forward(index, 60.)
        progress = planner._progress_at(index, ego)
        transition = d.get('transition')
        lane = transition['target'] if transition else d['current_lane']
        planner.transition = transition
        path = planner._path(ahead, ego, lane, progress, transition)
        a = [obstacle(o) for o in scenes[scene_index][1] if not o.get('road_surface')]
        b = [obstacle(o) for o in scenes[scene_index][2] if not o.get('road_surface')]
        lead_a = planner._lead(path, ego, a, 55.)
        lead_b = planner._lead(path, ego, b, 55., lane=lane)
        evaluated += 1
        if lead_a is not None and lead_b is None:
            changes.append(dict(stamp=d['stamp'], lane=lane, transition=bool(transition),
                old_distance=lead_a[0], old_box=dict(x=lead_a[1].x, y=lead_a[1].y,
                    length=lead_a[1].length, width=lead_a[1].width)))
        without_labels = planner._lead(path, ego, b, 55.)
        if without_labels is not None and lead_b is None:
            stable_ignored += 1
    return dict(run=name, recorded_scenes=len(scenes), evaluated_highway_scenes=evaluated,
        narrowed_coasting_boxes=len(coasting), coasting_examples=coasting[:8],
        lane_labels=[dict(raceline=k[0], stable=k[1], boundary_crossing=k[2], count=v)
                     for k, v in sorted(labels.items())],
        measured_width_jumps_over_half_metre=len(measured_growth),
        measured_growth_examples=measured_growth[:8],
        leader_removed_scenes=len(changes), stable_neighbour_margin_exclusions=stable_ignored,
        leader_changes=changes[:12],
        recorded_brake_reasons=[dict(reason=k[0], state=k[1], count=v)
                                for k, v in brakes.most_common()],
        limitation='Recorded-pose geometry comparison, not a new closed-loop drive.')


if __name__ == '__main__':
    result = [analyse(name) for name in ('run_20261018_225612', 'run_20261018_225727')]
    (Path(__file__).parent/'box_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in result:
        print(json.dumps({k: v for k, v in row.items() if not isinstance(v, list)}))
