"""Recorded ego/obstacles, no vehicle simulation or ROS outputs."""
from pathlib import Path
import sys,importlib.util,json,math,yaml
import rosbag
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
h=module(ROOT/'analysis/2026-10-22_highway_side_vehicle/replay_lane_change.py','helper')
from avoidance_runtime import SETTINGS
old=module(ROOT/'rollback/2026-10-28_s11_merge_timed_decel_before/src/unita_waypoint/scripts/local_avoidance_core.py','old')
new=module(ROOT/'src/unita_waypoint/scripts/local_avoidance_core.py','new')
report=[]
for r in [ROOT/'src/unita_waypoint/log'/name for name in ('run_20261028_205505','run_20261028_205548')]:
 planners=[]
 params=yaml.safe_load((r/'rosparams.yaml').read_text())['gps_waypoint_follower']
 for core in (old,new):
  p,route,sectors,speeds=h.build(core,r/'config/map_tunner_project.json')
  for name,spec in SETTINGS.items():
   if spec[0]=='gps_waypoint_follower' and name in params:setattr(p,spec[2],params[name])
  planners.append(p)
 results=[dict(first_r2=None,first_r3=None,first_slow=None,reaccel=None,recovery=None,strong_frames=0,staged_frames=0,targets=[],waits={}),dict(first_r2=None,first_r3=None,first_slow=None,reaccel=None,recovery=None,strong_frames=0,staged_frames=0,targets=[],waits={})]
 last=-1.;stamp=None
 for bp in sorted(r.glob('*.bag')):
  with rosbag.Bag(str(bp)) as b:
   start=b.get_start_time()
   for topic,m,t in b.read_messages(['/waypoint_debug/obstacles','/waypoint_debug/control_decision']):
    d=json.loads(m.data)
    if topic.endswith('/obstacles'):
     stamp=d['stamp'];continue
    dt=t.to_sec()-start;i=d.get('waypoint_index',0)
    if dt-last<.099 or sectors[i]!=11 or d.get('x') is None:continue
    last=dt
    tangent=(route[min(i+1,len(route)-1)][0]-route[i][0],route[min(i+1,len(route)-1)][1]-route[i][1])
    for core,p,result in zip((old,new),planners,results):
     obs=[core.Obstacle(**{('track_id' if k=='id' else k):v for k,v in o.items() if ('track_id' if k=='id' else k) in core.Obstacle.__dataclass_fields__}) for o in d['obstacles']]
     decision=p.plan(i,(d['x'],d['y']),d['current_speed_kmh']/3.6,obs,ego_yaw=math.atan2(tangent[1],tangent[0]),desired_speed_mps=speeds[i]/3.6,observation_stamp=stamp)
     result['targets'].append((round(dt,2),decision.speed_cap_kmh))
     tr=p.transition
     if tr and tr['target'] in (2,3):
      key='first_r'+str(tr['target'])
      if result[key] is None:result[key]=dict(t=round(dt,2),x=round(d['x'],2),y=round(d['y'],2),target=decision.speed_cap_kmh)
     if decision.speed_cap_kmh <= 30. and result['first_slow'] is None:
      result['first_slow']=dict(t=round(dt,2),y=round(d['y'],2),speed=round(d['current_speed_kmh'],2),reason=decision.reason)
     if result['first_slow'] is not None and result['reaccel'] is None and dt>12.5 and decision.speed_cap_kmh > d['current_speed_kmh']+.5:
      result['reaccel']=dict(t=round(dt,2),y=round(d['y'],2),target=decision.speed_cap_kmh)
     if 13.<=dt<=16.5 and decision.speed_cap_kmh>=d['current_speed_kmh']+5.:
      result['strong_frames']+=1
     if result['recovery'] is None and dt>12.5 and d['current_speed_kmh']<40. and decision.speed_cap_kmh >= d['current_speed_kmh']+5.:
      result['recovery']=dict(t=round(dt,2),target=decision.speed_cap_kmh)
     if decision.reason=='s11_r1_merge_yield' and decision.speed_cap_kmh>35. and d['y']>140.:
      result['staged_frames']+=1
     if decision.reason.startswith(('s11_wait','s11_r1_merge')):
      result['waits'][decision.reason]=result['waits'].get(decision.reason,0)+1
 pairs=zip(results[0].pop('targets'),results[1].pop('targets'))
 diffs=[dict(t=a[0],old=a[1],new=b[1]) for a,b in pairs if a[1]!=b[1]]
 item=dict(run=r.name,old=results[0],new=results[1],changed_frames=len(diffs),examples=diffs[:8]);report.append(item);print(json.dumps(item),flush=True)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(dict(limitation='Open-loop replay at recorded ego poses, sampled about 10 Hz; route tangent approximates ego yaw. Not collision or closed-loop validation.',runs=report),indent=2))
