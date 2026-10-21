"""Frozen pre-fix versus current ground-reference API on recorded LiDAR.

Only S14 receives the override. S15 extension is reported separately as hypothetical.
No ROS node construction, publishers, vehicle commands, or source edits.
"""
from pathlib import Path
import bisect,hashlib,importlib.util,json,sys
import numpy as np
import rosbag,yaml
from leo_avoidance.ftg_local import Config,FollowTheGap
from leo_guard.route_file import load_route_points
from leo_guard.traffic_local import RouteProgress
ROOT=Path('/home/unita/unita_ws');OUT=Path(__file__).resolve().parent;LOG=ROOT/'src/unita_waypoint/log';RUN=LOG/'run_20261021_230042'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
baseline=module('s14_low_box_filter_baseline',OUT/'ftg_point_filter_baseline.py')
node=module('s14_low_box_scope_node',ROOT/'src/leo_avoidance/scripts/driving_state_node_local.py')
params=yaml.safe_load((RUN/'rosparams.yaml').read_text())['leo_driving_state'];old=baseline.FollowTheGap(baseline.Config(**{k:params.get(k,v) for k,v in vars(baseline.Config()).items()}));new=FollowTheGap(Config(**{k:params.get(k,v) for k,v in vars(Config()).items()}))
route=RouteProgress(load_route_points(str(RUN/'config/map_tunner_project.json')),max_error=7.,backward_window=30)
bounds=node.s14_ground_interval(str(RUN/'config/map_tunner_project.json'),route)
scope=node.DrivingStateNode.__new__(node.DrivingStateNode);scope.s14_ground_bounds=bounds;scope.s14_ground_reference_z_m=-1.61
prior=json.loads((OUT/'low_point_stage_evidence.json').read_text());prior_rows={r['stamp']:r for r in prior['rows']}
poses=[];tracking=[]
with rosbag.Bag(str(LOG/'semi_leo_2026-10-21-23-00-42.bag')) as b:
 start=b.get_start_time()
 for t,m,s in b.read_messages(topics=['/leo/driving_status','/leo/waypoint_debug/tracking_state']):
  (poses if t.endswith('driving_status') else tracking).append((s.to_sec(),json.loads(m.data)))
def nearest(seq,stamp):
 i=bisect.bisect_left([x[0] for x in seq],stamp);return min(seq[max(0,i-1):i+1],key=lambda x:abs(x[0]-stamp))[1]
def xyz(m):
 off={f.name:f.offset for f in m.fields};dt=np.dtype({'names':['x','y','z'],'formats':['<f4']*3,'offsets':[off[k] for k in ['x','y','z']],'itemsize':m.point_step});v=np.frombuffer(m.data,dtype=dt,count=m.width*m.height);a=np.column_stack([v[k] for k in ['x','y','z']]).astype(float);return a[np.isfinite(a).all(axis=1)]
def old_stages(p):
 a=[p];a.append(old._remove_ground(a[-1]));a.append(old._in_collision_height(a[-1]));a.append(old._remove_ego_returns(a[-1]));a.append(old._in_avoidance_sector(a[-1]));a.append(old._track_obstacles(a[-1])[0]);return a

def new_stages(p,ref):
 ground,plane=new._ground_filter(p,ref);a=[p,ground];a.append(new._in_collision_height(a[-1],ref,plane));a.append(new._remove_ego_returns(a[-1]));a.append(new._in_avoidance_sector(a[-1]));a.append(new._track_obstacles(a[-1])[0]);return a,plane

def roi_counts(stages,c):
 lo=np.array(c['lidar_min_xyz'])-np.array([.05,.05,.01]);hi=np.array(c['lidar_max_xyz'])+np.array([.05,.05,.01]);return [int(((p>=lo)&(p<=hi)).all(axis=1).sum()) for p in stages]
source_path=ROOT/'src/leo_avoidance/src/leo_avoidance/ftg_local.py';source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
stats={'total_scans':0,'default_api_exact_all_stages_scans':0,'s14_applied_scans':0,'outside_s14_scans':0,'outside_s14_exact_all_stages_scans':0,'s15_hypothetical_scans':0,'s14_valid_plane_scans':0,'scope_vs_recorded_sector_disagreements':[]};counts={};rows=[];examples=[]
with rosbag.Bag(str(RUN/'sensors_2026-10-21-23-00-42_0.bag')) as b:
 for _,m,t in b.read_messages(topics=['/lidar3D']):
  stamp=m.header.stamp.to_sec();sec=stamp-start;p=xyz(m);recorded=nearest(tracking,stamp);pose=nearest(poses,stamp).get('driving_pose');sector=recorded['sector'];progress=None
  try:progress,_=node.locate_or_reacquire(route,*pose);reference=scope.ground_reference_for_progress(progress)
  except (TypeError,ValueError):reference=None
  active=reference is not None;original=old_stages(p);default,_=new_stages(p,None)
  assert all(np.array_equal(a,b) for a,b in zip(original,default)),('default changed',stamp)
  stats['total_scans']+=1;stats['default_api_exact_all_stages_scans']+=1
  applied,plane=new_stages(p,reference) if active else (default,None)
  if active:
   stats['s14_applied_scans']+=1;stats['s14_valid_plane_scans']+=int(plane is not None)
  else:
   stats['outside_s14_scans']+=1
   assert all(np.array_equal(a,b) for a,b in zip(original,applied)),('outside changed',stamp)
   stats['outside_s14_exact_all_stages_scans']+=1
  if active!=(sector==14):stats['scope_vs_recorded_sector_disagreements'].append({'stamp':stamp,'sector':sector,'route_progress_m':progress,'active':active})
  hypothetical=None
  if sector==15:
   hypothetical,_=new_stages(p,-1.61);stats['s15_hypothetical_scans']+=1
  key='S14_actual' if active else ('S15_default' if sector==15 else 'other_default')
  acc=counts.setdefault(key,{'scans':0,'original':np.zeros(6,dtype=int),'applied':np.zeros(6,dtype=int)})
  acc['scans']+=1;acc['original']+=[len(a) for a in original];acc['applied']+=[len(a) for a in applied]
  if hypothetical is not None:
   acc=counts.setdefault('S15_hypothetical_extension',{'scans':0,'original':np.zeros(6,dtype=int),'applied':np.zeros(6,dtype=int)});acc['scans']+=1;acc['original']+=[len(a) for a in original];acc['applied']+=[len(a) for a in hypothetical]
  row={'stamp':stamp,'relative_s':sec,'recorded_sector':sector,'route_progress_m':progress,'s14_profile_applied':active,'original_counts':[len(a) for a in original],'applied_counts':[len(a) for a in applied],'s15_hypothetical_counts':[len(a) for a in hypothetical] if hypothetical is not None else None,'plane':plane.tolist() if plane is not None else None};rows.append(row)
  previous=prior_rows.get(stamp)
  selected=(previous is not None and any(abs(sec-q)<.015 for q in (27.0519,45.03965,45.97572,46.99473,47.46875,48.02477)))
  if selected:
   # Public APIs must agree with the independently stepped production pipeline.
   assert np.array_equal(new.obstacle_candidates(p,ground_reference_z_m=reference),applied[-1])
   assert np.array_equal(new.rear_obstacle_candidates(p,ground_reference_z_m=reference),new._rear_obstacle_points(applied[3]))
   candidates=[]
   for c in previous['low_candidates']:
    x,y=c['approx_map_xy'];entry=abs(x-2.25)<1.5 and abs(y+508.11)<1.5;exitbox=abs(x+84.3)<1.5 and abs(y+554)<1.0;pitched=sec>48 and 10<c['lidar_median_xyz'][0]<12
    if entry or exitbox or pitched:
     candidates.append({'kind':'entry_left' if entry else 'exit_box' if exitbox else 'pitched_11m_low_returns','xyz':c['lidar_median_xyz'],'height_above_road':c['height_above_fitted_road_percentiles'],'original_stage_roi_counts':roi_counts(original,c),'s14_only_applied_stage_roi_counts':roi_counts(applied,c),'s15_hypothetical_stage_roi_counts':roi_counts(hypothetical,c) if hypothetical is not None else None})
   examples.append({**row,'candidates':candidates})
  if stats['total_scans']%100==0:print('scans',stats['total_scans'],flush=True)
for v in counts.values():
 for k in ('original','applied'):v[k]=v[k].tolist()
result={'source_filter_sha256':source_hash,'s14_interval_s_m':bounds,'stages':prior['stages'],'stats':stats,'point_totals':counts,'candidate_examples':examples,'rows':rows,'scope':'Applied branch uses the new exact CP13<=route-progress<CP14 helper on recorded capture-time-nearest driving pose. S15 always receives None; its -1.61 override appears only in the separately labelled hypothetical extension.','limits':['Per-frame filtering only, not a closed-loop vehicle simulation or full tracker-state replay.','Recorded pose/status nearest to capture time is used; original internal odometry callback ordering is not reconstructed.','No live ROS nodes, production code edits, or changes to the scenario.']}
(OUT/'final_ground_filter_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'stats':stats,'counts':counts,'examples':examples},indent=2))
