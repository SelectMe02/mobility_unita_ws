"""Recorded published obstacle/pose reconstruction; no control output."""
import rosbag,json
from leo_avoidance.route_avoidance import RouteAvoidance,RouteFrame
from map_tunner_core import primary_route
from pathlib import Path
project=json.loads(Path('src/unita_waypoint/config/map_extreme_project.json').read_text())
points,_=primary_route(project,raceline1_only=True)
xy=[(p['x'],p['y']) for p in points]
import runpy
p=runpy.run_path('src/leo_avoidance/test/test_route_avoidance.py')['hold_planner']()
route=RouteFrame(xy);state=None;speed=0.;result=[]
with rosbag.Bag('src/unita_waypoint/log/semi_leo_2026-10-30-01-07-38.bag') as b:
 start=b.get_start_time()
 for topic,m,t in b.read_messages(topics=['/leo/route_avoidance','/leo/waypoint_debug/tracking_state','/competition/ego_speed_kmh']):
  sec=t.to_sec()-start
  if topic.endswith('tracking_state'):state=json.loads(m.data);continue
  if topic.endswith('ego_speed_kmh'):speed=m.data/3.6;continue
  if not 208.<sec<211.63 or not state:continue
  d=json.loads(m.data);p.observation_time=sec
  for o in d['obstacles']:
   h=p.s4_motion_history.setdefault(o['id'],[])
   h.append((sec,o['s'],o['d'],o['l'],o['r'],o['moving']))
   h[:]=[v for v in h if sec-v[0]<=1.]
   if o['id']!=231:continue
   s,_,_=route.project(state['x'],state['y'],0,len(route.lengths))
   front=s+p.config.front_extent_m
   early=p._interrupted_crossing_approach(o,speed,front)
   result.append(dict(t=sec,d=o['d'],new_early=bool(early),recorded_cap=d['speed_cap_kmh']))
output=dict(first_new_evidence=next((v['t'] for v in result if v['new_early']),None),
 first_recorded_cap=next((v['t'] for v in result if v['recorded_cap'] is not None),None),
 limitation='Rounded published obstacle coordinates and received-time pose; not raw tracker replay or closed-loop stopping verification.',samples=result)
Path('analysis/2026-10-30_early_crossing/replay.json').write_text(json.dumps(output,indent=2))
print({k:v for k,v in output.items() if k!='samples'})
