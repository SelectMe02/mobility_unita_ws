import json,sys,math
from pathlib import Path
import numpy as np
import rosbag
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
from map_tunner_core import load_project,primary_route

def distance(point,line):
 a=line[:-1];v=np.diff(line,axis=0);vv=(v*v).sum(axis=1)
 t=np.clip(((np.asarray(point)-a)*v).sum(axis=1)/np.maximum(vv,1e-12),0,1)
 return float(np.linalg.norm(np.asarray(point)-(a+t[:,None]*v),axis=1).min())

reports=[]
for run in sorted((ROOT/'src/unita_waypoint/log').glob('run_20261028_170*')):
 project=load_project(str(run/'config/map_tunner_project.json'))
 points,sectors=primary_route(project);route=np.array([(p['x'],p['y']) for p in points]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1))]
 lanes={int(n):np.array([(p['x'],p['y']) for p in pts]) for sec in project['sectors'] if sec['number']==11 for n,pts in sec['racelines'].items() if pts}
 rows=[];path=[];speed=0
 for bp in run.glob('*.bag'):
  with rosbag.Bag(str(bp)) as bag:
   t0=bag.get_start_time()
   for topic,m,ts in bag.read_messages(['/waypoint_debug/local_path','/waypoint_debug/tracking_state','/competition/ego_speed_kmh']):
    if topic.endswith('local_path'):path=[(p.pose.position.x,p.pose.position.y) for p in m.poses];continue
    if topic.endswith('ego_speed_kmh'):speed=m.data;continue
    d=json.loads(m.data)
    if d.get('sector')!=11 or not d.get('transition'):continue
    tr=d['transition'];i=d['waypoint_index'];ego=(d['x'],d['y']);a=route[i];v=route[min(i+1,len(route)-1)]-a;v/=max(np.linalg.norm(v),1e-6)
    progress=float(arc[i]+(np.array(ego)-a)@v)
    target=tr['target'];line=lanes[target];pursuit=d.get('pursuit_target')
    row=dict(t=round(ts.to_sec()-t0,3),source=tr['source'],target=target,length=tr['length'],fraction=(progress-tr['start'])/tr['length'],speed=speed,
     ego_error=distance(ego,line),lookahead=d['lookahead_m'],pursuit_error=distance((pursuit['x'],pursuit['y']),line) if pursuit else None,
     path_end_error=distance(path[-1],line) if path else None,
     lane_errors={n:distance(ego,l) for n,l in lanes.items()})
    rows.append(row)
 print(run.name,'frames',len(rows))
 for pair in sorted(set((r['source'],r['target']) for r in rows)):
  group=[r for r in rows if (r['source'],r['target'])==pair]
  late=[r for r in group if r['fraction']>=1]
  print(pair,'time',group[0]['t'],group[-1]['t'],'length',round(group[0]['length'],2),'late',len(late))
  for label,subset in [('all',group),('after_planned_end',late)]:
   if subset:print(label,{key:[round(float(np.min([r[key] for r in subset])),3),round(float(np.median([r[key] for r in subset])),3),round(float(np.max([r[key] for r in subset])),3)] for key in ('ego_error','pursuit_error','path_end_error','lookahead')})
  for r in group[::max(1,len(group)//6)]:print({k:round(v,3) if isinstance(v,float) else v for k,v in r.items()})
 reports.append(dict(run=run.name,rows=rows))
(Path(__file__).parent/'evidence.json').write_text(json.dumps(reports,indent=2))
