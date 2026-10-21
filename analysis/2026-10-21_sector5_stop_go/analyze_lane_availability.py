"""Read-only S5 lane availability from the run's map snapshot and actual planner."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path[:0] = [str(ROOT / p) for p in (
    'src/unita_waypoint/scripts', 'src/leo_guard/src', 'src/leo_avoidance/src')]
from map_tunner_core import load_project, primary_route, saved_lane_indices
from local_avoidance_core_local import LocalAvoidancePlanner
from leo_guard.route_file import load_lane_changes
from leo_avoidance.route_avoidance import RouteAvoidance

snapshot = ROOT / 'src/unita_waypoint/log/run_20261019_002529/config/map_tunner_project.json'
project = load_project(snapshot)
route, sectors = primary_route(project, raceline1_only=True)
xy = [(float(p['x']), float(p['y'])) for p in route]
indices = saved_lane_indices(project, route, sectors)
lanes = {n: [(float(p['x']), float(p['y'])) for sector in project['sectors']
             for p in sector['racelines'].get(str(n), [])] for n in (2, 3)}
planner = LocalAvoidancePlanner(xy, sectors, lanes, lane_indices=indices,
    min_turn_radius_m=3.0 / math.tan(math.radians(29)),
    pass_speed_kmh=20.0, r1_tracking_tolerance_m=0.85)
permissions = load_lane_changes(ROOT / 'src/leo_avoidance/config/route_lane_changes.json',
                                np.asarray(xy))
route_planner = RouteAvoidance(xy, lane_changes=permissions)
sector = next(s for s in project['sectors'] if s['number'] == 5)
r2 = sector['racelines']['2']
gaps = [dict(before=a, after=b, source_id_difference=b['source_id']-a['source_id'],
             endpoint_distance_m=math.hypot(b['x']-a['x'], b['y']-a['y']))
        for a, b in zip(r2, r2[1:]) if b['source_id']-a['source_id'] > 1]

rows = []
for index in (800, 846, 856, 857, 864, 870, 877, 881, 903, 911, 924, 931, 932, 949):
    point = xy[index]
    r2_point = planner._lane_point(2, point, index)
    available = [n for n in (1, 2, 3)
                 if planner._lane_point(n, point, index) is not None]
    span = planner._span(2, planner._forward(index, 60.0))
    s = route_planner.route.s[index]
    rows.append(dict(route_wp_index=index, map_point_id=route[index]['id'],
        route_s_m=float(s), x=point[0], y=point[1], available_lanes=available,
        r2_point=r2_point, r2_contiguous_span_m=span,
        required_span_for_change_at_20kmh_m=(planner._merge_length(point, r2_point, 20/3.6)+8
                                           if r2_point else None),
        route_permission_for_next_20m=route_planner.pass_offsets(s, s+20)))

observations = []
for row in json.loads((OUT / 'leo_waypoint_debug_tracking_state.json').read_text()):
    data = row['data']
    if data.get('sector') != 5 or data.get('driving_state') != 'TAILING':
        continue
    index = data['waypoint_index']
    observations.append(dict(t=row['t'], route_wp_index=index, map_point_id=route[index]['id'],
        y=data['y'], kind=data.get('obstacle_kind'), distance=data.get('obstacle_distance_m'),
        r2_available=planner._lane_point(2, xy[index], index) is not None,
        r2_contiguous_span_m=planner._span(2, planner._forward(index, 60.))))

result = dict(snapshot=str(snapshot.relative_to(ROOT)),
    sector_lane_count=sector['lane_count'], saved_r1_count=len(sector['racelines']['1']),
    saved_r2_count=len(r2), r2_gaps=gaps,
    missing_r2_route_indices=[857, 931], missing_r2_map_source_id_bounds=[871, 947],
    note='Map point IDs and zero-based deduplicated route waypoint indices are different.',
    projected_route_lane_permissions=[p for p in permissions if p['start_s'] < 550],
    planner_rows=rows, actual_tailing_observations=observations,
    source_references=dict(
        loader='gps_waypoint_follower_local.py HEAD:c3cc13f lines 630-650; constructor 478-480',
        mapping='map_tunner_core.py:253-285, S5 maps exact source_id; only S11 interpolates',
        lookup='local_avoidance_core_local.py:159-164 returns mapping.get(index)',
        availability='local_avoidance_core_local.py:444-446 removes missing lane points',
        span='local_avoidance_core_local.py:215-221 stops at first missing mapped point',
        change_rejection='local_avoidance_core_local.py:534-536 requires merge length plus 8m',
        route_permission='route_avoidance.py:254-267 requires entire pass in allowed intervals',
        route_wait='route_avoidance.py:330-376 returns WAIT_SOLID_LINE when offsets empty',
        local_fallback='local_avoidance_core_local.py:600-609 returns TAILING for lead obstacle'))
(OUT / 'lane_availability.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'actual_tailing_observations'}, indent=2))
print('Actual S5 TAILING observations:', len(observations),
      'with R2 available:', sum(o['r2_available'] for o in observations))
