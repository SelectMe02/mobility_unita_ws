"""Replay recorded tunnel clouds through FTG and log left/right wall fits.

Read-only: no ROS publishers. Used to check that both fitted walls are
continuously present inside the tunnel and disappear together at the portal.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rosbag
import yaml
from sensor_msgs import point_cloud2
from leo_avoidance.ftg_local import Config, FollowTheGap

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'src/unita_waypoint/log'


def run(leo_bag, run_dir, t_from, t_to):
    params = yaml.safe_load((LOG / run_dir / 'rosparams.yaml').read_text())['leo_driving_state']
    config = Config(**{k: params.get(k, v) for k, v in vars(Config()).items()})
    clock = [0.]
    ftg = FollowTheGap(config, clock=lambda: clock[0])
    captured = {}
    original = ftg._track_obstacles

    def spy(points):
        obstacles, walls = original(points)
        captured['walls'] = walls
        return obstacles, walls
    ftg._track_obstacles = spy
    with rosbag.Bag(str(LOG / leo_bag)) as bag:
        t0 = bag.get_start_time()
        speeds = [(ts.to_sec(), m.data / 3.6) for _, m, ts in
                  bag.read_messages(['/competition/ego_speed_kmh'])]
        modes = [(ts.to_sec(), json.loads(m.data)['mode']) for _, m, ts in
                 bag.read_messages(['/leo/driving_status'])]
    st = np.array([s[0] for s in speeds]); mt = np.array([m[0] for m in modes])
    rows = []
    for path in sorted((LOG / run_dir).glob('sensors_*.bag')):
        with rosbag.Bag(str(path)) as bag:
            for _, msg, ts in bag.read_messages(['/lidar3D']):
                t = ts.to_sec() - t0
                if not t_from <= t <= t_to:
                    continue
                clock[0] = ts.to_sec()
                pts = np.asarray(list(point_cloud2.read_points(
                    msg, field_names=('x', 'y', 'z'), skip_nans=True)), float).reshape(-1, 3)
                speed = speeds[int(np.argmin(abs(st - ts.to_sec())))][1]
                mode = modes[int(np.argmin(abs(mt - ts.to_sec())))][1]
                captured.clear()
                ftg.plan(pts, speed, 0., wall_follow=True, wall_follow_weight=1.,
                         ego_pose=(0., 0., 0.), stamp=msg.header.stamp.to_sec())
                walls = captured.get('walls', (None, None))
                desc = []
                for w in walls:
                    desc.append(None if w is None else
                                (round(float(w[0]), 2), round(float(w[5]), 1), round(float(w[6]), 1)))
                rows.append((round(t, 2), mode, round(speed * 3.6, 1), len(pts), desc[0], desc[1]))
    return rows


if __name__ == '__main__':
    leo_bag, run_dir, a, b = sys.argv[1:5]
    for row in run(leo_bag, run_dir, float(a), float(b)):
        print(*row)
