"""Offline recorded traffic + path-following ego approximation; no ROS output.

NPC detections follow the original bag timeline, not the newly driven ego.
Overlap checks cover confirmed car-sized tracks and are not a safety proof.
"""
import bisect
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/unita_waypoint/scripts'))
sys.path.insert(0, str(ROOT / 'src/unita_waypoint/test'))
from map_tunner_core import load_project, primary_route, checkpoint_route_indices, saved_lane_indices
from local_avoidance_core import LocalAvoidancePlanner, Obstacle
from gps_waypoint_follower import GPSWaypointFollower
from test_highway_planner import point_at

project = load_project(str(ROOT / 'src/unita_waypoint/config/map_tunner_project.json'))
points, sectors, route_lanes = primary_route(project, with_lanes=True)
route = [(float(q['x']), float(q['y'])) for q in points]
speeds = [float(q['speed_kmh']) for q in points]
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
planner = LocalAvoidancePlanner(route, sectors, lanes, lane_indices=indices,
                                route_lanes=route_lanes, preferred_lane=1,
                                preferred_lane_indices=preferred, finish_line=finish,
                                highway_sectors=(11,), sector_min_speed_kmh=30.)
follower = GPSWaypointFollower.__new__(GPSWaypointFollower)
follower.local_planner = planner
follower.route_sectors = sectors
follower.target_speed_for_index = lambda index: speeds[index]
run_name = sys.argv[1] if len(sys.argv) > 1 else 'run_20261018_215638'
recorded = json.loads((Path(__file__).parent / (run_name+'_comparison.json')).read_text())
frames = [(row['stamp'], dict(row, waypoint_index=row['index'], current_speed_kmh=row['speed_kmh']))
          for row in recorded]
frames.sort(key=lambda item: item[0])
stamps = [s for s, _ in frames]
t0, initial = next((s, d) for s, d in frames
                   if checkpoints['10'] <= (d.get('waypoint_index') or 0) < end)
index = initial['waypoint_index']
ego = (initial['x'], initial['y'])
speed = (initial['current_speed_kmh'] or 0.) / 3.6
dt, t = .05, 0.
minimum_speed = speed
cruise_seen = speed >= 30./3.6
minimum_after_cruise = math.inf
zero_caps = overlaps = 0
changes, last_key = [], None
while t < 60.:
    index = min(range(max(0, index-5), min(len(route), index+40)),
                key=lambda i: math.dist(route[i], ego))
    if sectors[index] != 11 and index > checkpoints['10']+5:
        break
    frame = frames[min(len(frames)-1, bisect.bisect_left(stamps, t0+t))][1]
    obstacles = [Obstacle(**{key: o[key] for key in (
        'x', 'y', 'radius', 'vx', 'vy', 'kind', 'confirmed', 'velocity_reliable',
        'length', 'width', 'yaw', 'road_surface')}) for o in frame['obstacles']]
    a, b = route[index], route[index+1]
    norm = math.dist(a, b)
    tx, ty = (b[0]-a[0])/norm, (b[1]-a[1])/norm
    desired = speeds[index]/3.6
    decision = planner.plan(index, ego, speed, obstacles,
                            ego_yaw=math.atan2(ty, tx), desired_speed_mps=desired)
    zero_caps += decision.speed_cap_kmh <= .01
    target = min(desired, decision.speed_cap_kmh/3.6)
    follower.current_waypoint_index = index
    follower.current_speed_kmh = speed*3.6
    follower.last_decision = decision
    target = follower.sector_speed_floor(target*3.6)/3.6
    speed = max(0., speed+max(-6.*dt, min(1.4*dt, target-speed)))
    minimum_speed = min(minimum_speed, speed)
    cruise_seen = cruise_seen or speed >= 30./3.6
    if cruise_seen:
        minimum_after_cruise = min(minimum_after_cruise, speed)
    key = decision.state, decision.reason, planner.current_lane
    if key != last_key:
        changes.append(dict(time_s=round(t, 2), index=index, lane=planner.current_lane,
                            state=decision.state, reason=decision.reason,
                            speed_kmh=speed*3.6, cap_kmh=decision.speed_cap_kmh))
        last_key = key
    ego = (point_at(decision.path, speed*dt) if len(decision.path) >= 2
           else (ego[0]+tx*speed*dt, ego[1]+ty*speed*dt))
    for obstacle in obstacles:
        if not obstacle.confirmed or obstacle.length < 2.5:
            continue
        dx, dy = obstacle.x-ego[0], obstacle.y-ego[1]
        lon, lat = dx*tx+dy*ty, -dx*ty+dy*tx
        if (-.79-obstacle.length/2 < lon < 3.85+obstacle.length/2
                and abs(lat) < .95+obstacle.width/2):
            overlaps += 1
    t += dt
report = dict(end_index=index, cp11_index=end, final_lane=planner.current_lane,
              elapsed_s=round(t, 2), minimum_speed_kmh=minimum_speed*3.6,
              minimum_after_reaching_30_kmh=minimum_after_cruise*3.6,
              zero_cap_samples=zero_caps, confirmed_car_overlap_samples=overlaps,
              changes=changes)
(Path(__file__).parent / (run_name+'_replay.json')).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'changes'}, indent=2))
assert index >= end and planner.current_lane == 3 and overlaps == 0
