"""Offline per-frame S12 lane-mask replay; no node construction or publishers."""
import bisect, collections, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import rosbag
import yaml
ROOT=Path('/home/unita/unita_ws')
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
from lane_corridor import LaneCorridor, SectorLaneFilter
from lidar_obstacles_local import road_obstacle_mask
from leo_avoidance.road_points import obstacle_points,RoadPointConfig
from leo_avoidance.route_avoidance import RouteAvoidance
from leo_guard.route_file import load_route_points
sensor=ROOT/'src/unita_waypoint/log/run_20261021_220130/sensors_2026-10-21-22-01-31_0.bag'
semi=ROOT/'src/unita_waypoint/log/semi_leo_2026-10-21-22-01-30.bag'
project_file=ROOT/'src/unita_waypoint/log/run_20261021_220130/config/map_tunner_project.json'
params=yaml.safe_load((sensor.parent/'rosparams.yaml').read_text())
poses={};geometry=None
for source,topic in [(sensor,'/localization/pose'),(semi,'/leo/driving_pose')]:
 values={}
 with rosbag.Bag(str(source)) as bag:
  for name,m,t in bag.read_messages(topics=[topic,'/waypoint_debug/lane_geometry']):
   if name.endswith('lane_geometry'):
    geometry=json.loads(m.data);continue
   p,q=m.pose.position,m.pose.orientation
   values[m.header.stamp.to_sec()]=(p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
 ordered=sorted(values.items());poses[topic]=([t for t,p in ordered],[p for t,p in ordered])
assert geometry is not None
geometry_copy=json.dumps(geometry,sort_keys=True)
original=LaneCorridor(geometry);scoped=SectorLaneFilter(geometry,sector=12,width=3.2)
assert json.dumps(geometry,sort_keys=True)==geometry_copy
spec=importlib.util.spec_from_file_location('route_node_mask_verifier',ROOT/'src/leo_avoidance/scripts/route_avoidance_node.py')
route_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(route_module)
route_node=route_module.RouteAvoidanceNode.__new__(route_module.RouteAvoidanceNode)
route_node.lidar_to_rear_x_m=float(params['leo_route_avoidance'].get('lidar_to_rear_x_m',1.4))
route_node.planner=RouteAvoidance(load_route_points(str(project_file))[:,:2])
rpc=params['leo_route_avoidance'];defaults=RoadPointConfig(lidar_to_rear_x_m=route_node.lidar_to_rear_x_m)
road_cfg=RoadPointConfig(**{key:tuple(rpc.get('road_'+key,val)) if isinstance(val,tuple) else float(rpc.get('road_'+key,val)) for key,val in vars(defaults).items()})
lp=params['leo_lidar_obstacles']

def pose_at(topic,stamp):
 stamps,values=poses[topic];k=bisect.bisect_left(stamps,stamp)
 if k<len(stamps) and abs(stamps[k]-stamp)<1e-6:return values[k]
 if k==0 or k==len(stamps):return None
 a,b=stamps[k-1],stamps[k]
 if b-a>.15:return None
 f=(stamp-a)/(b-a);p,q=values[k-1],values[k];dyaw=math.atan2(math.sin(q[2]-p[2]),math.cos(q[2]-p[2]))
 return p[0]+f*(q[0]-p[0]),p[1]+f*(q[1]-p[1]),p[2]+f*dyaw

def xyz(message):
 offsets={f.name:f.offset for f in message.fields};endian='>' if message.is_bigendian else '<'
 dtype=np.dtype({'names':['x','y','z'],'formats':[endian+'f4']*3,'offsets':[offsets[k] for k in ['x','y','z']],'itemsize':message.point_step})
 a=np.frombuffer(message.data,dtype=dtype,count=message.width*message.height)
 p=np.column_stack((a['x'],a['y'],a['z'])).astype(float)
 return p[np.isfinite(p).all(axis=1)]

def map_xy(p,pose,lidar_x=0.):
 c,s=math.cos(pose[2]),math.sin(pose[2]);x=p[:,0]+lidar_x;y=p[:,1]
 return np.column_stack((pose[0]+c*x-s*y,pose[1]+s*x+c*y))

stats={};rows=[]
def record(name,stamp,pose,before,after,active):
 d=stats.setdefault(name,dict(frames=0,s12_frames=0,outside_frames=0,before_points_s12=0,after_points_s12=0,outside_changed_frames=0,s12_first_stamp=None,s12_last_stamp=None,max_removed_single_frame=0))
 d['frames']+=1;d['s12_frames' if active else 'outside_frames']+=1
 changed=not np.array_equal(before,after)
 if active:
  d['before_points_s12']+=len(before);d['after_points_s12']+=len(after)
  d['s12_first_stamp']=stamp if d['s12_first_stamp'] is None else min(stamp,d['s12_first_stamp']);d['s12_last_stamp']=stamp if d['s12_last_stamp'] is None else max(stamp,d['s12_last_stamp'])
  d['max_removed_single_frame']=max(d['max_removed_single_frame'],len(before)-len(after))
 else:d['outside_changed_frames']+=int(changed)
 if active:rows.append(dict(source=name,stamp=stamp,ego_xy=list(pose[:2]),before_count=len(before),after_count=len(after)))

missing=collections.Counter();count=0
with rosbag.Bag(str(sensor)) as bag:
 for topic,m,t in bag.read_messages(topics=['/lidar3D','/waypoint_debug/obstacle_points','/waypoint_debug/obstacles']):
  if topic.endswith('/obstacles'):
   d=json.loads(m.data);stamp=float(d['stamp']);pose=pose_at('/localization/pose',stamp)
   if pose is None:missing['main_obstacles']+=1;continue
   before=np.array([[o['x'],o['y']] for o in d['obstacles']],float).reshape(-1,2);active=scoped.active(pose[:2]);after=before[scoped.corridor.contains(before)] if active else before
   record('main_logged_obstacle_centers',stamp,pose,before,after,active);continue
  stamp=m.header.stamp.to_sec()
  if topic.endswith('obstacle_points'):
   pose=pose_at('/localization/pose',stamp)
   if pose is None:missing['main_debug_points']+=1;continue
   assert m.header.frame_id=='map'
   before=xyz(m);active=scoped.active(pose[:2]);after=before[scoped.corridor.contains(before[:,:2])] if active else before
   record('main_recorded_used_points',stamp,pose,before,after,active);continue
  p=xyz(m)
  pose=pose_at('/localization/pose',stamp)
  if pose is None:missing['local_lidar_raw']+=1
  else:
   corrected=p+np.array([float(lp.get('lidar_to_rear_x_m',1.4)),float(lp.get('lidar_to_rear_y_m',0.)),float(lp.get('lidar_to_rear_z_m',1.23))])
   mask=road_obstacle_mask(corrected,float(lp.get('front_range_m',55.)),float(lp.get('rear_range_m',55.)),max(float(lp.get('side_range_m',12.)),3*original.width),float(lp.get('vehicle_half_width_m',.946)),float(lp.get('front_extent_m',3.845)),float(lp.get('rear_extent_m',.79)))
   candidates=corrected[mask];mapped=map_xy(candidates,pose)
   before=candidates[original.contains(mapped)];active=scoped.active(pose[:2]);after=candidates[scoped.corridor.contains(mapped)] if active else before
   record('local_lidar_road_and_lane_points',stamp,pose,before,after,active)
  pose=pose_at('/leo/driving_pose',stamp)
  if pose is None:missing['route_raw']+=1
  else:
   road=obstacle_points(p,road_cfg)
   active=scoped.active(pose[:2]);mapped=map_xy(road,pose,route_node.lidar_to_rear_x_m)
   lane=road[scoped.corridor.contains(mapped)] if active else road
   record('route_ground_filtered_points_lane_mask',stamp,pose,road,lane,active)
   before=route_node.drop_road_boundaries(road,pose)
   after=route_node.drop_road_boundaries(lane,pose) if active else before
   record('route_final_boundary_filtered_points',stamp,pose,before,after,active)
   route_node.planner._locate_ego(*pose[:2])
  count+=1
  if count%300==0:print('processed raw scans',count,flush=True)
with rosbag.Bag(str(semi)) as bag:
 for topic,m,t in bag.read_messages(topics=['/leo/waypoint_debug/obstacles']):
  d=json.loads(m.data);stamp=float(d['stamp']);pose=pose_at('/localization/pose',stamp)
  if pose is None:missing['local_obstacles']+=1;continue
  before=np.array([[o['x'],o['y']] for o in d['obstacles']],float).reshape(-1,2);active=scoped.active(pose[:2]);after=before[scoped.corridor.contains(before)] if active else before
  record('local_logged_obstacle_centers',stamp,pose,before,after,active)

lanes={v['sector']:np.array(v['points'],float) for v in geometry['lanes'] if v['raceline']==1}
q=lanes[12];i=len(q)//2;mid=.5*(q[i]+q[i+1]);tan=q[i+1]-q[i];tan/=np.linalg.norm(tan);normal=np.array([-tan[1],tan[0]])
edge=[]
for offset in [0.,1.5,1.6,1.7,-1.5,-1.6,-1.7]:
 pt=mid+offset*normal;edge.append(dict(offset_m=offset,old_contains=bool(original.contains([pt])[0]),new_contains=bool(scoped.corridor.contains([pt])[0])))
assert all(r['new_contains']==(abs(r['offset_m'])<=1.6) for r in edge)
end=q[-1];endtan=q[-1]-q[-2];endtan/=np.linalg.norm(endtan)
bridge=.5*(lanes[11][-1]+q[0])
boundaries={'s12_first_active':scoped.active(q[0]),'s12_last_active':scoped.active(q[-1]),'incoming_long_bridge_active':scoped.active(bridge),'just_before_s12_last_active':scoped.active(end-.01*endtan),'s13_first_in_perception_corridor':bool(scoped.corridor.contains([lanes[13][0]])[0]),'s13_10th_in_perception_corridor':bool(scoped.corridor.contains([lanes[13][10]])[0])}
assert boundaries=={'s12_first_active':True,'s12_last_active':False,'incoming_long_bridge_active':False,'just_before_s12_last_active':True,'s13_first_in_perception_corridor':True,'s13_10th_in_perception_corridor':True}
assert all(d['outside_changed_frames']==0 for d in stats.values())
result={'source_bags':[str(sensor),str(semi)],'map_sha256':hashlib.sha256(project_file.read_bytes()).hexdigest(),'recorded_geometry_width_m':original.width,'s12_width_m':scoped.corridor.width,'method':'Read-only per-frame mask replay using capture-time interpolated pose. Local lidar reproduces its road ROI and existing union-lane mask; route reproduces obstacle_points, applies the new lane mask, then runs existing boundary removal, matching production order; the original route pipeline applies only boundary removal; main lidar starts at its recorded used-point cloud (so ground/IMU segmentation is not rerun). Recorded obstacle centers are filtered independently without rerunning tracker state. Outside S12 arrays are compared exactly.','limitations':['Offline recorded-data analysis only; no live ROS nodes or vehicle commands were used, and this is not real-vehicle driving validation.','This verifies spatial filtering per recorded frame, not vehicle closed-loop behavior or complete post-change tracking histories.','Missing bracketing poses (>0.15 s gaps or outside recorded range) are skipped and counted.','Route and local poses are interpolated from their own recorded pose topics; callback arrival ordering is not reproduced.'],'stats':stats,'skipped_pose_frames':dict(missing),'real_map_edge_examples':edge,'real_map_boundary_checks':boundaries,'s12_frames':rows}
out=Path(__file__).resolve().with_name('s12_bag_mask_replay.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('stats','skipped_pose_frames','real_map_edge_examples','real_map_boundary_checks')},ensure_ascii=False,indent=2));print(out)
