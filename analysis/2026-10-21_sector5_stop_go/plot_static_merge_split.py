from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

base = Path(__file__).parent
records = json.loads((base / 'static_id96_raw_evidence.json').read_text())
clouds = np.load(base / 'static_id96_raw_points.npz')
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True, sharey=True)
selected = [records[0], records[3], records[4]]
titles = ['54.711: two separate clusters', '55.058: merged cluster, ID 99',
          '55.187: split + ID switch, cap 0']
for ax, record, title in zip(axes, selected, titles):
    points = clouds[str(round(record['cloud_stamp'], 6))]
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], vmin=.3, vmax=2.5,
               cmap='viridis', s=14)
    for group in record['reclustered_groups']:
        low, high = np.array(group['min_map']), np.array(group['max_map'])
        cx = group['center_map'][0]
        track = min(record['recorded_tracks'], key=lambda o: abs(o['x'] - cx))
        ax.add_patch(Rectangle(low[:2], high[0]-low[0], high[1]-low[1],
                               fill=False, edgecolor='#d14b23', linewidth=1.6))
        ax.text(cx, high[1]+.08, f"ID {track['id']}", ha='center',
                fontsize=10, color='#992b0a')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-58.2, -55.8)
    ax.set_ylim(-25.3, -23.4)
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    ax.set_xlabel('Map X (m)')
axes[0].set_ylabel('Map Y (m)')
fig.suptitle('S5 obstacle flicker: stationary returns merge and split', fontsize=14)
fig.text(.5, .025, 'Raw LiDAR; rectangles show cluster bounds. All times are 00:27:ss.sss KST, October 19.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .08, 1, .94])
fig.savefig(base / 'static_id96_merge_split.png', dpi=180)
