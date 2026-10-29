import rosbag,json,sys,math,importlib.util
import numpy as np
from sensor_msgs import point_cloud2
from leo_avoidance.route_avoidance import RouteFrame,RouteAvoidanceConfig
from leo_avoidance.observed_clearance import pedestrian_clearance_observer
from map_tunner_core import primary_route
p=json.load(open('src/unita_waypoint/config/map_extreme_project.json'));pts,_=primary_route(p,raceline1_only=True);route=RouteFrame([(x['x'],x['y']) for x in pts]);cfg=RouteAvoidanceConfig()
name='oldclear';spec=importlib.util.spec_from_file_location(name,'rollback/2026-10-30_s5_observed_corridor_release_before/src/leo_avoidance/src/leo_avoidance/observed_clearance.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
bag='src/unita_waypoint/log/semi_leo_2026-10-29-23-31-52.bag'
with rosbag.Bag(bag) as b:start=b.get_start_time()
h=dict(s=459.6,d=-2.14,l=.38)
pose=(-60.04095713474089,-20.293234440498054,1.59710)
import runpy
helpers=runpy.run_path('src/leo_avoidance/test/test_route_avoidance.py')
planner=helpers['hold_planner']();hold=helpers['candidate_boundary_hold']()
hold.update(ids={51},s=459.6,d=-2.14,l=.38,last_l=.38,last_r=.29,
            vd=2.6,seen_at=76.74,velocity_at=76.74,stationary_scans=0)
planner.pedestrian_hold=hold
with rosbag.Bag(bag) as source:
 observations=[(t.to_sec()-start,json.loads(m.data)['obstacles']) for _,m,t in source.read_messages(topics=['/leo/route_avoidance'])]
counts=[0,0,0];release=None;rows=[]
with rosbag.Bag('src/unita_waypoint/log/run_20261029_233153/sensors_2026-10-29-23-31-53_0.bag') as b:
 for _,m,t in b.read_messages(topics=['/lidar3D']):
  sec=t.to_sec()-start
  if not 80<sec<90:continue
  raw=np.array(list(point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)))
  a=old.pedestrian_clearance_observer(raw,pose,route,cfg,1.4)(h,lambda s:0.)
  c=pedestrian_clearance_observer(raw,pose,route,cfg,1.4)(h,lambda s:0.)
  counts[0]+=1;counts[1]+=a;counts[2]+=c
  items=next((o for time,o in reversed(observations) if time<=sec),[])
  if release is None:
   planner.observation_time=sec
   planner.clearance_observer=lambda hold,offset: c
   cap=planner._single_pedestrian_hold_cap(items,0.,451.013643,454.858643,lambda s:0.)
   if cap is None:release=sec
  rows.append(dict(t=sec,old_clear=bool(a),new_clear=bool(c)))
print('scans, old clear, new clear',counts)

result=dict(scans=counts[0],old_clear=counts[1],new_clear=counts[2],release_at_s=release,
 limitations=['Reconstructed rounded hold and stopped pose, recorded raw scans and route obstacles from 80-90 seconds.','Tests orphan-hold release only, not closed-loop vehicle acceleration or the complete original planner replay.'],rows=rows)
json.dump(result,open('analysis/2026-10-30_s5_observed_release/replay.json','w'),indent=2)
print('orphan release',release)
