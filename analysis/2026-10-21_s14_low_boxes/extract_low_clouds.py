from pathlib import Path
import json,math,bisect,dataclasses,importlib.util,sys
import numpy as np
import rosbag,yaml
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
spec=importlib.util.spec_from_file_location('low_box_ftg_baseline',Path(__file__).resolve().with_name('ftg_point_filter_baseline.py'))
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
Config,FollowTheGap=module.Config,module.FollowTheGap
ROOT=Path('/home/unita/unita_ws');F=ROOT/'src/unita_waypoint/log/run_20261021_230042';OUT=Path(__file__).resolve().parent
params=yaml.safe_load((F/'rosparams.yaml').read_text())['leo_driving_state'];cfg=Config(**{k:params.get(k,v) for k,v in vars(Config()).items()});ftg=FollowTheGap(cfg)
changed=FollowTheGap(dataclasses.replace(cfg,expected_ground_z_m=-1.61,collision_min_z_m=-1.51))
semi=ROOT/'src/unita_waypoint/log/semi_leo_2026-10-21-23-00-42.bag';poses=[];states=[];plans=[]
with rosbag.Bag(str(semi)) as bag:
 start=bag.get_start_time()
 for t,m,s in bag.read_messages(topics=['/leo/driving_status','/leo/waypoint_debug/tracking_state','/leo/avoidance_status']):
  d=json.loads(m.data)
  if t.endswith('driving_status'):poses.append((s.to_sec(),d))
  elif t.endswith('tracking_state'):states.append((s.to_sec(),d))
  else:plans.append((s.to_sec(),d))
def near(seq,t):
 i=bisect.bisect_left([x[0] for x in seq],t);return min(seq[max(0,i-1):i+1],key=lambda x:abs(x[0]-t))[1]
def xyz(m):
 off={f.name:f.offset for f in m.fields};dt=np.dtype({'names':['x','y','z'],'formats':['<f4']*3,'offsets':[off[k] for k in ['x','y','z']],'itemsize':m.point_step});a=np.frombuffer(m.data,dtype=dt,count=m.width*m.height);p=np.column_stack([a[k] for k in ['x','y','z']]).astype(float);return p[np.isfinite(p).all(axis=1)]
def estimate(p):
 sample=p[(p[:,0]>=1)&(p[:,0]<=12)&(abs(p[:,1])<=5)&(p[:,2]>=-2.5)&(p[:,2]<=-.3)]
 if len(sample)<60:return None
 low=sample[sample[:,2]<=np.percentile(sample[:,2],40)];u,n=np.unique(np.floor(low[:,2]/.1).astype(int),return_counts=True);z=(u[np.argmax(n)]+.5)*.1;seed=sample[abs(sample[:,2]-z)<=.18]
 for _ in range(2):
  if len(seed)<30 or np.ptp(seed[:,0])<2 or np.ptp(seed[:,1])<2:return None
  plane=np.linalg.lstsq(np.column_stack([seed[:,:2],np.ones(len(seed))]),seed[:,2],rcond=None)[0]
  seed=sample[abs(sample[:,2]-sample[:,:2]@plane[:2]-plane[2])<=.1]
 return plane

def groups(points):
 if len(points)<3:return []
 tree=cKDTree(points[:,:2]);pairs=tree.query_pairs(.35);parent=np.arange(len(points))
 def root(i):
  while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
  return i
 for a,b in pairs:
  a,b=root(a),root(b)
  if a!=b:parent[a]=b
 ids=np.array([root(i) for i in range(len(points))]);out=[]
 for i in np.unique(ids):
  q=points[ids==i]
  if len(q)<5:continue
  if max(np.ptp(q[:,0]),np.ptp(q[:,1]))>2.5:continue
  out.append(q)
 return out
rows=[];selected={};targets=[20.,21.,22.,23.,24.,26.,40.,42.,43.,44.,45.,46.,47.,48.]
with rosbag.Bag(str(F/'sensors_2026-10-21-23-00-42_0.bag')) as bag:
 for _,m,s in bag.read_messages(topics=['/lidar3D']):
  stamp=m.header.stamp.to_sec();sec=stamp-start
  if not 17<=sec<=50:continue
  p=xyz(m);state=near(states,stamp);drive=near(poses,stamp);plan=near(plans,stamp);pose=drive.get('driving_pose');plane=estimate(p)
  stage=[p];stage.append(ftg._remove_ground(stage[-1]));stage.append(ftg._in_collision_height(stage[-1]));stage.append(ftg._remove_ego_returns(stage[-1]));stage.append(ftg._in_avoidance_sector(stage[-1]));stage.append(ftg._track_obstacles(stage[-1])[0])
  changedstage=[p];changedstage.append(changed._remove_ground(changedstage[-1]));changedstage.append(changed._in_collision_height(changedstage[-1]));changedstage.append(changed._remove_ego_returns(changedstage[-1]));changedstage.append(changed._in_avoidance_sector(changedstage[-1]));changedstage.append(changed._track_obstacles(changedstage[-1])[0])
  candidates=[]
  if plane is not None:
   h=p[:,2]-p[:,:2]@plane[:2]-plane[2]
   mask=(h>.1)&(h<.55)&(p[:,2]<cfg.collision_min_z_m)&(p[:,0]>=0)&(p[:,0]<=25)&(abs(p[:,1])<5)
   for q in groups(p[mask]):
    center=np.median(q,axis=0);res=q[:,2]-q[:,:2]@plane[:2]-plane[2]
    def inside(v):return (v[:,0]>=q[:,0].min()-.05)&(v[:,0]<=q[:,0].max()+.05)&(v[:,1]>=q[:,1].min()-.05)&(v[:,1]<=q[:,1].max()+.05)&(v[:,2]>=q[:,2].min()-.01)&(v[:,2]<=q[:,2].max()+.01)
    world=None
    if pose:
     x=center[0]+cfg.lidar_to_rear_x_m;y=center[1];c,si=math.cos(pose[2]),math.sin(pose[2]);world=[pose[0]+c*x-si*y,pose[1]+si*x+c*y]
    candidates.append({'raw_points':len(q),'lidar_median_xyz':center.tolist(),'lidar_min_xyz':q.min(axis=0).tolist(),'lidar_max_xyz':q.max(axis=0).tolist(),'height_above_fitted_road_percentiles':np.percentile(res,[0,50,100]).tolist(),'approx_map_xy':world,'original_stage_roi_counts':[int(inside(v).sum()) for v in stage],'lowered_reference_stage_roi_counts':[int(inside(v).sum()) for v in changedstage]})
  rows.append({'stamp':stamp,'relative_s':sec,'sector':state['sector'],'ego':pose,'mode':drive['mode'],'ftg_phase':plan['avoidance_phase'],'front_points':plan['front_points'],'bearing_deg':math.degrees(plan['target_bearing_rad']),'plane_diagnostic':plane.tolist() if plane is not None else None,'original_stage_counts':[len(v) for v in stage],'lowered_reference_stage_counts':[len(v) for v in changedstage],'low_candidates':candidates})
  for target in targets:
   if target not in selected or abs(sec-target)<abs(selected[target][0]-target):selected[target]=(sec,p,plane,state['sector'])
fig,axs=plt.subplots(7,2,figsize=(16,27))
for ax,(target,(sec,p,plane,sector)) in zip(axs.flat,selected.items()):
 if plane is None:continue
 h=p[:,2]-p[:,:2]@plane[:2]-plane[2];mask=(p[:,0]>=-3)&(p[:,0]<28)&(abs(p[:,1])<5)&(h>.07)&(h<1.)
 ax.scatter(p[mask,0],p[mask,1],c=h[mask],cmap='turbo',vmin=.0,vmax=.6,s=6);ax.set_xlim(-3,28);ax.set_ylim(-5,5);ax.axhline(0,color='gray');ax.grid(True);ax.set_title('%.3fs sector %d plane z %.3f'%(sec,sector,plane[2]));ax.set_aspect('equal');ax.set_xlabel('LiDAR x forward (m)');ax.set_ylabel('LiDAR y left (m)')
fig.tight_layout();fig.savefig(OUT/'low_return_clouds.png',dpi=120)
result={'source_bag':str(F/'sensors_2026-10-21-23-00-42_0.bag'),'stages':['raw','ground','height','ego','sector','wall'],'original_ground_reference':cfg.expected_ground_z_m,'original_collision_min_z':cfg.collision_min_z_m,'comparison_ground_reference':-1.61,'comparison_collision_min_z':-1.51,'limits':['Offline point filtering only; no source modifications or live ROS.','FTG source is not included in run snapshot. Ground/height/ego/wall code is frozen in ftg_point_filter_baseline.py from the workspace at analysis start; earlier heading fix only is present in that copy.','Diagnostic plane reproduces the ground fit but does not enforce the configured absolute reference gate; low candidates are geometric hypotheses, not manually identified boxes.','Map positions use LiDAR odometry pose and are approximate; no simulator truth is recorded.','Reference comparison is diagnostic, not an approved code change.'], 'rows':rows};(OUT/'low_point_stage_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print('wrote',len(rows),'frames',OUT)
