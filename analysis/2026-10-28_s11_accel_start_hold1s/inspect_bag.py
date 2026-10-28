"""Extract recorded control commands; this does not simulate the revised vehicle."""
import json
from pathlib import Path
import rosbag
ROOT = Path(__file__).resolve().parents[2]
run = ROOT/'src/unita_waypoint/log/run_20261028_185236'
rows = []
for path in sorted(run.glob('*.bag')):
    with rosbag.Bag(str(path)) as bag:
        start = bag.get_start_time()
        for _, msg, stamp in bag.read_messages(['/waypoint_debug/control_decision']):
            d = json.loads(msg.data)
            rows.append(dict(t=round(stamp.to_sec()-start, 3), **{
                k:d.get(k) for k in ('waypoint_index','current_speed_kmh','target_speed_kmh',
                'command','state','current_lane','transition','lane_change_reason','speed_limits')}))
(Path(__file__).parent/'evidence.json').write_text(json.dumps(dict(
    source=str(run.relative_to(ROOT)), limitation='Recorded commands only; revised code requires a new MORAI run.',
    rows=rows), indent=2))
previous = None
for row in rows:
    state = (row['state'], row['current_lane'])
    if state != previous:
        print(json.dumps(row));previous=state
