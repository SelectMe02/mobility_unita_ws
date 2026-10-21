"""Summarize recorded control ownership, low-box candidates, and exit impact."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,bisect,math
import numpy as np
import rosbag
ROOT=Path('/home/unita/unita_ws');OUT=Path(__file__).resolve().parent;LOG=ROOT/'src/unita_waypoint/log';rows=json.loads((OUT/'low_point_stage_evidence.json').read_text())['rows'];semi=LOG/'semi_leo_2026-10-21-23-00-42.bag';sensor=LOG/'run_20261021_230042/sensors_2026-10-21-23-00-42_0.bag';raw=[];other=[];cmds=[];states=[];modes=[]
with rosbag.Bag(str(semi)) as b:
 start=b.get_start_time()
 for topic,m,t in b.read_messages(topics=['/leo/raw_ctrl_cmd','/semi/raceline_ctrl_cmd','/ctrl_cmd','/leo/waypoint_debug/tracking_state','/leo/driving_status']):
  stamp=t.to_sec()
  if hasattr(m,'data'):
   (states if topic.endswith('tracking_state') else modes).append((stamp,json.loads(m.data)));continue
  values=[m.steering,m.accel,m.brake,m.longlCmdType,m.velocity]
  if topic=='/leo/raw_ctrl_cmd':raw.append((stamp,values))
  elif topic=='/semi/raceline_ctrl_cmd':other.append((stamp,values))
  else:cmds.append((stamp,values))
def nearest(seq,stamp):
 i=bisect.bisect_left([x[0] for x in seq],stamp);return min(seq[max(0,i-1):i+1],key=lambda x:abs(x[0]-stamp))[1]
owner={'blackout_ctrl_samples':0,'matched_recent_leo_raw':0,'matched_recent_raceline':0}
for t,c in cmds:
 if nearest(modes,t)['mode']!='BLACKOUT':continue
 owner['blackout_ctrl_samples']+=1
 for source,key in ((raw,'matched_recent_leo_raw'),(other,'matched_recent_raceline')):
  i=bisect.bisect_right([x[0] for x in source],t);owner[key]+=int(any(c==x[1] and 0<=t-x[0]<=.15 for x in source[max(0,i-4):i]))
impacts=[]
with rosbag.Bag(str(sensor)) as b:
 for _,m,t in b.read_messages(topics=['/imu']):
  stamp=t.to_sec();sec=stamp-start
  if not 40<=sec<=50:continue
  if abs(m.linear_acceleration.z-9.81)<3 and abs(m.linear_acceleration.x)<4:continue
  q=m.orientation;state=nearest(states,stamp);drive=nearest(modes,stamp)
  impacts.append({'stamp':stamp,'time':datetime.fromtimestamp(stamp,timezone(timedelta(hours=9))).isoformat(timespec='milliseconds'),'relative_s':sec,'sector':state['sector'],'waypoint_index':state['waypoint_index'],'map_pose':drive['driving_pose'],'drive_mode':drive['mode'],'acceleration_xyz_mps2':[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],'pitch_deg':math.degrees(math.asin(np.clip(2*(q.w*q.y-q.z*q.x),-1,1))),'pitch_rate_rad_s':m.angular_velocity.y})
examples=[]
for target,kind in [(27.052,'entry_left_candidate'),(45.040,'exit_candidate_seen_in_s14'),(45.976,'exit_candidate_seen_in_s14'),(46.995,'exit_candidate_in_s15'),(47.469,'exit_candidate_in_s15')]:
 row=min(rows,key=lambda x:abs(x['relative_s']-target));candidates=[]
 for c in row['low_candidates']:
  x,y=c['approx_map_xy']
  if (kind.startswith('entry') and abs(x-2.25)<1.5 and abs(y+508.11)<1.5) or (kind.startswith('exit') and abs(x+84.3)<1.5 and abs(y+554)<1.):candidates.append(c)
 examples.append({'kind':kind,'stamp':row['stamp'],'relative_s':row['relative_s'],'sector':row['sector'],'ego':row['ego'],'plane':row['plane_diagnostic'],'candidates':candidates})
stats={}
for sector in (14,15):
 a=[r for r in rows if r['sector']==sector];planes=[r['plane_diagnostic'][2] for r in a if r['plane_diagnostic']]
 stats[str(sector)]={'scans':len(a),'old_ground_fit_removes_points_scans':sum(r['original_stage_counts'][0]!=r['original_stage_counts'][1] for r in a),'new_reference_ground_fit_removes_points_scans':sum(r['lowered_reference_stage_counts'][0]!=r['lowered_reference_stage_counts'][1] for r in a),'diagnostic_plane_intercept_min_median_max_m':np.percentile(planes,[0,50,100]).tolist()}
result={'source_bags':[str(semi),str(sensor)],'control_owner':owner,'ground_reference_stats':stats,'candidate_examples':examples,'exit_impulse_events':impacts,'conclusion':'Recorded low-box candidate points survive the old ground stage but are discarded by absolute height threshold -1.13 m. Measured S14 road intercept median is -1.617 m; expected ground -1.23 m rejects every observed S14 plane. The in-memory -1.61/-1.51 reference profile retains entry/exit candidate points with the existing 0.1 m road-plane threshold.','scope':'Exit impact candidate at +47.922 s is in S15, though the same stationary low return cluster is already visible from S14 at +45.040 s. S15 code changes require user scope authorization.','limits':['Box labels are inferred from compact low-return geometry and user report; no simulator object labels or collision event topic were recorded.','IMU impulse supports contact but does not independently identify the contacted object.','Reference-profile comparison is point-level offline diagnosis, not closed-loop safety proof; no production code edits were made.']}
(OUT/'diagnosis_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'owner':owner,'ground':stats,'impact':impacts},indent=2))
