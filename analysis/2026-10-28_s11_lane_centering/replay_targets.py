"""Compare recorded pursuit targets with corrected paths at the SAME ego poses."""
import importlib.util,json,math,sys
from pathlib import Path
import rosbag
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
import local_avoidance_core as core
spec=importlib.util.spec_from_file_location('helper',ROOT/'analysis/2026-10-22_highway_side_vehicle/replay_lane_change.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
report={'limitation':'Same-pose geometry comparison only, not closed-loop MORAI validation.','runs':[]}
for run in sorted((ROOT/'src/unita_waypoint/log').glob('run_20261028_170*')):
 p,route,sectors,speeds=h.build(core,run/'config/map_tunner_project.json')
 rows=[];missing=0
 for bp in run.glob('*.bag'):
  with rosbag.Bag(str(bp)) as bag:
   t0=bag.get_start_time()
   for _,m,ts in bag.read_messages(['/waypoint_debug/tracking_state']):
    d=json.loads(m.data);tr=d.get('transition')
    if d.get('sector')!=11 or not tr or not tr.get('s11_forward_only'):continue
    i=d['waypoint_index'];ego=(d['x'],d['y']);progress=p._progress_at(i,ego)
    if progress<tr['start']+tr['length']:continue
    ahead=p._forward(i,80.)
    path=p._path(ahead,ego,tr['target'],progress,tr)
    center=[]
    for j,pt,_ in ahead:
     v=p._lane_point(tr['target'],pt,j)
     if v is None:break
     center.append(v)
    if len(center)<2:missing+=1;continue
    yaw=d['yaw'];chosen=None
    for x,y in path:
     dx,dy=x-ego[0],y-ego[1]
     if math.cos(yaw)*dx+math.sin(yaw)*dy>0 and math.hypot(dx,dy)>=d['lookahead_m']:
      chosen=(x,y);break
    if chosen is None or not d.get('pursuit_target'):missing+=1;continue
    old=d['pursuit_target']
    before=core.project_polyline((old['x'],old['y']),center)[0]
    after=core.project_polyline(chosen,center)[0]
    assert after<1e-8
    rows.append(dict(t=round(ts.to_sec()-t0,3),old_error_m=before,new_error_m=after,
                     target_lane=tr['target'],lookahead_m=d['lookahead_m']))
 stats=dict(run=run.name,compared_frames=len(rows),missing_target_geometry=missing,
            old_median_m=float(np.median([r['old_error_m'] for r in rows])),
            new_max_m=max(r['new_error_m'] for r in rows),rows=rows)
 report['runs'].append(stats)
 print({k:v for k,v in stats.items() if k!='rows'},flush=True)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(report,indent=2))
