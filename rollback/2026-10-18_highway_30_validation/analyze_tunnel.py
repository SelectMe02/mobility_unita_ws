"""Read-only tunnel diagnostics from recorded ROS bags; publishes nothing."""
import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import rosbag

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).parent
reports = []
all_rows = {}
for name in ('semi_leo_2026-10-18-21-56-38.bag', 'semi_leo_2026-10-18-22-04-39.bag'):
    mode, status, steering = None, {}, 0.
    rows, switches, rejoins = [], [], []
    with rosbag.Bag(str(ROOT / 'src/unita_waypoint/log' / name)) as bag:
        for topic, message, stamp in bag.read_messages(topics=[
                '/leo/driving_mode', '/leo/driving_status', '/leo/avoidance_status', '/ctrl_cmd']):
            t = stamp.to_sec()
            if topic == '/leo/driving_mode':
                if message.data != mode:
                    switches.append((t, message.data))
                mode = message.data
            elif topic == '/ctrl_cmd':
                steering = message.steering
            elif topic == '/leo/driving_status':
                status = json.loads(message.data)
                if status.get('gps_rejoin_diagnostic'):
                    rejoins.append(dict(time=t, **status))
            elif mode == 'BLACKOUT':
                data = json.loads(message.data)
                rows.append(dict(time=t, steering_rad=steering,
                                 map_pose=status.get('map_pose'), route_index=status.get('blackout_route_index'),
                                 icp_rmse_m=status.get('icp_rmse_m'),
                                 driving_pose_jump_m=status.get('driving_pose_jump_m'), **data))
    if not rows:
        continue
    start = rows[0]['time']
    for row in rows:
        row['elapsed_s'] = row['time']-start
    def bounds(field, selected=rows):
        values = [r[field] for r in selected if r.get(field) is not None]
        return [float(x) for x in np.percentile(values, [0, 5, 50, 95, 100])] if values else None
    unique = [r for i, r in enumerate(rows) if i == 0 or
              (r['target_bearing_rad'], r.get('right_wall_clearance_m'), r['avoidance_phase']) !=
              (rows[i-1]['target_bearing_rad'], rows[i-1].get('right_wall_clearance_m'), rows[i-1]['avoidance_phase'])]
    jumps = sorted([dict(before_s=a['elapsed_s'], after_s=b['elapsed_s'],
                         before_phase=a['avoidance_phase'], after_phase=b['avoidance_phase'],
                         before_deg=math.degrees(a['target_bearing_rad']),
                         after_deg=math.degrees(b['target_bearing_rad']),
                         jump_deg=math.degrees(abs(a['target_bearing_rad']-b['target_bearing_rad'])),
                         front_points=b['front_points'], wall_clearance=b.get('right_wall_clearance_m'),
                         wall_heading_deg=(math.degrees(b['right_wall_heading_rad'])
                                           if b.get('right_wall_heading_rad') is not None else None),
                         rejoin_remaining=b.get('route_rejoin_remaining_m'))
                        for a, b in zip(unique, unique[1:])], key=lambda x: -x['jump_deg'])
    seconds = []
    for second in range(int(rows[-1]['elapsed_s'])+1):
        selected = [r for r in rows if second <= r['elapsed_s'] < second+1]
        if not selected:
            continue
        seconds.append(dict(second=second, phase=dict(Counter(r['avoidance_phase'] for r in selected)),
                            speed_kmh=np.mean([r['ego_speed_mps']*3.6 for r in selected]),
                            bearing_deg=[math.degrees(x) for x in bounds('target_bearing_rad', selected)[::2]],
                            wall_clearance=bounds('right_wall_clearance_m', selected),
                            wall_heading_deg=([math.degrees(x) for x in bounds('right_wall_heading_rad', selected)[::2]]
                                              if bounds('right_wall_heading_rad', selected) else None),
                            front_max=max(r['front_points'] for r in selected),
                            rejoin_remaining=next((r['route_rejoin_remaining_m'] for r in selected
                                                   if r.get('route_rejoin_remaining_m') is not None), None)))
    report = dict(file=name, switches=switches, duration_s=rows[-1]['elapsed_s'],
                  phases=dict(Counter(r['avoidance_phase'] for r in unique)),
                  phase_changes=sum(a['avoidance_phase'] != b['avoidance_phase'] for a, b in zip(unique, unique[1:])),
                  crossing_max=max(r['crossing_threats'] for r in rows),
                  speed_mps=bounds('ego_speed_mps'), wall_clearance_m=bounds('right_wall_clearance_m'),
                  wall_heading_rad=bounds('right_wall_heading_rad'), icp_rmse_m=bounds('icp_rmse_m'),
                  max_bearing_jumps=jumps[:12], seconds=seconds,
                  rejoin=next((r for r in rejoins if r.get('gps_rejoin_route_ok')), None))
    reports.append(report)
    all_rows[name] = rows
    print(json.dumps({k: v for k, v in report.items() if k not in ('seconds', 'rejoin')}, indent=2))
(OUTPUT/'tunnel_analysis.json').write_text(json.dumps(reports, indent=2)+'\n')
(OUTPUT/'tunnel_rows.json').write_text(json.dumps(all_rows)+'\n')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(reports), figsize=(13, 8), squeeze=False, sharex='col')
    for col, report in enumerate(reports):
        rows = all_rows[report['file']]
        x = [r['elapsed_s'] for r in rows]
        axes[0,col].plot(x, [math.degrees(r['target_bearing_rad']) for r in rows], label='FTG target')
        axes[0,col].plot(x, [math.degrees(r['steering_rad']) for r in rows], label='sent steering', alpha=.7)
        axes[1,col].plot(x, [r.get('right_wall_clearance_m') for r in rows], label='right-wall clearance')
        axes[1,col].axhline(2.9, color='gray', linestyle='--', label='spacing target')
        axes[2,col].plot(x, [r['ego_speed_mps']*3.6 for r in rows], label='ego speed')
        axes[0,col].set_title(report['file'].replace('semi_leo_2026-10-18-', '').replace('.bag',''))
        for ax in axes[:,col]:
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        axes[2,col].set_xlabel('Seconds since blackout')
    for i, label in enumerate(('Angle (degrees)', 'Clearance (m)', 'Speed (km/h)')):
        axes[i,0].set_ylabel(label)
    fig.tight_layout()
    fig.savefig(OUTPUT/'tunnel_analysis.png', dpi=160)
except ImportError:
    pass
