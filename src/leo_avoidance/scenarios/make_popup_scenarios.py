#!/usr/bin/env python3
"""Generate tunnel scenarios with static obstacles that appear and vanish.

MORAI waypoint events only offer speed changes and pauses, and no saved
scenario shows their JSON form. An obstacle therefore "appears" by rising
from below the road along its waypoint path once Ego is within
activeDistance, holds position on a short up-and-down path, and "vanishes"
by sinking below the road again. Waypoint-mode obstacles stop at the end of
their path, so it stays hidden afterwards.

Usage: make_popup_scenarios.py [output_scenario_dir]
Default output is the MORAI K-City scenario folder; the matching EgoNetwork
file is copied from the template scenario.
"""
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from leo_guard.route_file import load_route_points

HERE = Path(__file__).resolve().parent
ROUTE = HERE.parents[1] / 'unita_waypoint/config/map_tunner_project.json'
MORAI_DIR = Path.home() / ('MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/'
                           'Scenario/R_KR_PR_K-city_2025')
TEMPLATE = 'TUN03_car_lane1_departs_keeps_lane_plus_parked_car_lane2'
STATIC_OBSTACLE_ID = 40100052   # obstacle used by the blackout_obstacle scenarios
ROAD_Z = 28.42                  # road height inside the tunnel
LANE2_OFFSET_M = 3.5            # lane 2 is left of the lane 1 route
BURIED_M = 3.0                  # hidden depth below the road
RISE_SPEED_MPS = 4.0            # 3 m rise in 0.75 s
HOLD_BOB_M = 0.1                # hold path: road level <-> 10 cm lower

SCENARIOS = [
    {
        'name': 'TUN13_static_lane1_plus_lane2_popup_blocks_then_vanishes',
        'static': [{'route_s_m': 75., 'lateral_m': 0.}],
        'popup': [{'route_s_m': 78., 'lateral_m': LANE2_OFFSET_M,
                   'active_distance_m': 22., 'hold_s': 6.}],
    },
    {
        'name': 'TUN14_lane1_popup_then_vanishes_after_seconds',
        'static': [],
        'popup': [{'route_s_m': 90., 'lateral_m': 0.,
                   'active_distance_m': 25., 'hold_s': 4.}],
    },
]


def vec(x, y, z):
    return {'x': x, 'y': y, 'z': z, '_x': '%.3f' % x, '_y': '%.3f' % y,
            '_z': '%.3f' % z}


def unity(x, y, z):
    """Scenario ENU -> waypoint (Unity) coordinates, as in saved paths."""
    return vec(-x, z, -y)


def rot(yaw_deg):
    return {'roll': '0.000', 'pitch': '0.000', 'yaw': '%.3f' % yaw_deg}


class Route:
    def __init__(self, points, start_xy):
        self.points = points
        seg = np.diff(points, axis=0)
        self.lengths = np.linalg.norm(seg, axis=1)
        self.s = np.r_[0., np.cumsum(self.lengths)]
        start = int(np.argmin(np.linalg.norm(points - start_xy, axis=1)))
        self.s0 = self.s[start]

    def pose(self, route_s_m, lateral_m):
        """Point route_s_m along the route from Ego's start, offset left."""
        s = self.s0 + route_s_m
        i = int(np.searchsorted(self.s, s) - 1)
        while self.lengths[i] < 1e-6:
            i += 1
        d = (self.points[i + 1] - self.points[i]) / self.lengths[i]
        p = self.points[i] + d * (s - self.s[i])
        left = np.array([-d[1], d[0]])
        x, y = p + left * lateral_m
        return float(x), float(y), math.degrees(math.atan2(d[1], d[0]))


def obstacle(uid, x, y, z, yaw, waypoint_mode, speed=0., active=0.,
             init=None, events=0):
    return {
        'DataID': STATIC_OBSTACLE_ID, 'UNIQUEID': uid,
        'm_eobstacleObjType': 4, 'pos': vec(x, y, z), 'rot': rot(yaw),
        'scale': vec(1., 1., 1.),
        'initPos': init or vec(0., 0., 0.), 'initRot': rot(0.),
        'isWaypointMode': waypoint_mode, 'moveSpeed': speed,
        'activeDistance': active,
        'waypointEventList': [{'index': -1, 'waypointEventData': []}
                              for _ in range(events)],
    }


def popup_path(x, y, hold_s):
    hidden, top = ROAD_Z - BURIED_M, ROAD_Z
    heights = [hidden, top]
    bobs = max(1, int(round(RISE_SPEED_MPS * hold_s / HOLD_BOB_M)))
    for k in range(bobs):
        heights.append(top - HOLD_BOB_M if k % 2 == 0 else top)
    heights.append(hidden)
    return [{'pos': unity(x, y, z), 'rot': rot(0.)} for z in heights]


def build(template, route, spec):
    scenario = copy.deepcopy(template)
    objects, paths = [], []
    uid = 200
    for item in spec['static']:
        x, y, yaw = route.pose(item['route_s_m'], item['lateral_m'])
        objects.append(obstacle(uid, x, y, ROAD_Z, yaw, True))
        paths.append({'pathName': 'ObstaclePath - %d' % uid,
                      'waypointData': []})
        uid += 1
    for item in spec['popup']:
        x, y, yaw = route.pose(item['route_s_m'], item['lateral_m'])
        path = popup_path(x, y, item['hold_s'])
        objects.append(obstacle(
            uid, x, y, ROAD_Z - BURIED_M, yaw, True, RISE_SPEED_MPS,
            item['active_distance_m'], path[0]['pos'], len(path)))
        paths.append({'pathName': 'ObstaclePath - %d' % uid,
                      'waypointData': path})
        uid += 1
    scenario['objectList'] = objects
    scenario['waypointDataList'] = paths
    return scenario


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else MORAI_DIR
    template = json.load(open(MORAI_DIR / (TEMPLATE + '.json')))
    ego = template['egoVehicle']['initPosition']['pos']
    route = Route(load_route_points(ROUTE)[:, :2], np.array((ego['x'], ego['y'])))
    for spec in SCENARIOS:
        target = out / (spec['name'] + '.json')
        if target.exists():
            raise SystemExit('refusing to overwrite %s' % target)
        with open(target, 'w') as stream:
            json.dump(build(template, route, spec), stream, indent=4)
        network = out / 'EgoNetwork' / (spec['name'] + '_MN.json')
        source = MORAI_DIR / 'EgoNetwork' / (TEMPLATE + '_MN.json')
        if source.exists() and not network.exists():
            network.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, network)
        print(target)


if __name__ == '__main__':
    main()
