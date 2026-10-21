"""Plot recorded tunnel evidence; this is not a post-change vehicle run."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
data = json.loads((root / '224300_heading_evidence.json').read_text())
rows = [row for row in data['rows'] if row['avoidance_valid']
        and row['imu_yaw_deg'] is not None and row['wall_heading_deg'] is not None]
t = np.array([row['relative_s'] for row in rows])
yaw = np.array([row['imu_yaw_deg'] for row in rows])
wall = yaw + np.array([row['wall_heading_deg'] for row in rows])
reference = np.median(wall)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
axes[0].plot(t, yaw-reference, label='Vehicle heading (IMU)', color='#1565c0')
axes[0].plot(t, wall-reference, label='Wall direction in world frame', color='#388e3c')
axes[0].set_ylabel('Heading (deg)')
axes[0].legend(loc='upper left')
axes[1].plot(t, [row['bearing_deg'] for row in rows], label='FTG target bearing', color='#7b1fa2')
axes[1].plot(t, [row['steering_deg'] for row in rows], label='Published steering', color='#d84315')
axes[1].set_ylabel('Angle (deg)')
axes[1].legend(loc='upper left')
axes[2].plot(t, [row['wall_clearance_m'] for row in rows], color='#1565c0', label='Measured right-wall clearance')
axes[2].axhline(2.9, color='black', linestyle='--', label='Clearance target: 2.9 m')
axes[2].set_ylabel('Clearance (m)')
axes[2].set_xlabel('Seconds from bag start (2026-10-21 22:43:00.407 KST)')
axes[2].legend(loc='upper left')
for ax in axes:
    for row, following in zip(rows, rows[1:]):
        if row['phase'] == 'PASSING':
            ax.axvspan(row['relative_s'], following['relative_s'], color='#ffb300', alpha=.16, linewidth=0)
    ax.grid(alpha=.2)
fig.suptitle('Recorded S14 tunnel run before the fix — shaded: obstacle passing\n'
             'World wall direction is stable while vehicle heading oscillates')
fig.savefig(root / 'recorded_heading.png', dpi=150)
print(root / 'recorded_heading.png')
