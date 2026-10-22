"""Replay recorded exit inputs through the changed tunnel-exit decisions.

Uses recorded travelled distance, LiDAR route index, GPS poses/stamps and
Ego speed; right-wall spans come from wall_presence.py on the same clouds.
Not a closed-loop simulation: the recorded car still braked at the old HOLD.
"""
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import rosbag
import yaml
from leo_guard.route_file import anchor_on_route, load_route_points
from leo_guard.traffic_local import RouteProgress
from leo_avoidance.driving_state_local import DrivingState
from leo_avoidance.scan_odometry import ScanOdometry

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'src/unita_waypoint/log'
sys.path.insert(0, str(Path(__file__).parent))
from wall_presence import run as wall_rows

spec = importlib.util.spec_from_file_location(
    'exit_node', ROOT / 'src/leo_avoidance/scripts/driving_state_node_local.py')
node = importlib.util.module_from_spec(spec); spec.loader.exec_module(node)


def replay(leo_bag, run_dir, t_from, t_to):
    params = yaml.safe_load((LOG / run_dir / 'rosparams.yaml').read_text())['leo_driving_state']
    odo = yaml.safe_load((ROOT / 'src/leo_avoidance/config/odometry_local.yaml').read_text())
    points = load_route_points(str(LOG / run_dir / 'config/map_tunner_project.json'))
    s = node.DrivingStateNode.__new__(node.DrivingStateNode)
    s.machine = DrivingState(ScanOdometry(), max_blackout_distance_m=odo['max_blackout_distance_m'])
    s.machine.had_blackout = True
    s.route = RouteProgress(points, max_error=3., backward_window=30)
    s.blackout_route = RouteProgress(points, max_error=7., backward_window=30)
    s.blackout_end_index = anchor_on_route(points, params['blackout_end_anchor'])[0]
    s.exit_route_s = anchor_on_route(points, params['exit_anchor'])[1]
    for k in ('exit_portal_overrun_m', 'exit_tunnel_overrun_max_m', 'tunnel_wall_alongside_x_m',
              'tunnel_portal_detect_x_m', 'tunnel_portal_confirm_scans',
              'tunnel_portal_route_offset_m'):
        setattr(s, k, odo[k])
    s.reset_tunnel_wall()
    s.exit_rejoin_active = True       # recorded LiDAR progress was inside the window
    s.gps_history = deque(maxlen=100)
    s.speed_timeout_s = .35
    walls = {round(r[0], 2): r[4] for r in wall_rows(leo_bag, run_dir, t_from, t_to)}
    events = []
    with rosbag.Bag(str(LOG / leo_bag)) as bag:
        t0 = bag.get_start_time()
        for topic, m, ts in bag.read_messages(['/leo/driving_status', '/localization/pose',
                                               '/competition/ego_speed_kmh']):
            t = ts.to_sec() - t0
            if t_from <= t <= t_to:
                events.append((t, topic, m))
    wall_times = sorted(walls)
    wi = 0; first = {}; last_travelled = 0.
    for t, topic, m in events:
        while wi < len(wall_times) and wall_times[wi] <= t:
            s.machine.travelled_m = last_travelled
            span = walls[wall_times[wi]]
            s.update_tunnel_wall(None if span is None else (span[1], span[2]))
            wi += 1
        if topic == '/competition/ego_speed_kmh':
            s.latest_speed = (m.data / 3.6, t)
        elif topic == '/localization/pose':
            pose = (m.pose.position.x, m.pose.position.y, node.DrivingStateNode.yaw(m.pose.orientation))
            s.gps_history.append((pose, t, m.header.stamp.to_sec()))
            s.machine.update_raw_gps(37., 127., t)
            s.machine.update_gps_valid(True, t)
            s.machine.update_gps(pose, t)
        else:
            d = json.loads(m.data)
            last_travelled = d['travelled_m'] if d['travelled_m'] else last_travelled
            if d['blackout_route_index'] is not None and d['mode'] != 'GPS':
                s.route_progress_m = float(s.blackout_route.s[d['blackout_route_index']])
            if s.tunnel_portal_seen and 'portal latched' not in first:
                first['portal latched'] = round(t, 2)
            if d['mode'] == 'GPS' and 'GPS' not in first:
                first['recorded GPS'] = t
                break
            s.machine.travelled_m = last_travelled
            index = d['blackout_route_index']
            allowed = (index is not None and index <= s.blackout_end_index) or \
                s.tunnel_wall_extension_ok(s.blackout_route, index)
            if not allowed and 'new HOLD' not in first:
                first['new HOLD'] = (round(t, 2), round(last_travelled, 1), s.tunnel_wall_extent_m)
            if d['mode'] == 'HOLD' and 'recorded HOLD' not in first:
                first['recorded HOLD'] = round(t, 2)
            if s.machine.gps_pose is not None and s.machine.gps_return_since is None:
                pass
            if s.machine.gps_return_since is not None and 'new GPS accept' not in first:
                if s.exit_gps_rejoin_ok(t):
                    first['new GPS accept'] = round(t, 2)
            if s.machine.gps_pose is not None and 'GPS back' not in first:
                first['GPS back'] = round(t, 2)
                gp = s.machine.gps_pose[0]
                probe = RouteProgress(points, max_error=3.)
                gps_s, _ = probe.locate(*gp)
                anchored = s.portal_progress()
                first['anchored-GPS progress m'] = (None if anchored is None
                                                    else round(anchored - gps_s, 2))
                first['lidar-GPS progress m'] = round(s.route_progress_m - gps_s, 1)
    return first


if __name__ == '__main__':
    print(replay(*sys.argv[1:3], float(sys.argv[3]), float(sys.argv[4])))
