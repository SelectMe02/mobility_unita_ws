"""Read-only recorded-data analysis. Does not initialize ROS or command a vehicle."""
from pathlib import Path
import bisect,collections,json,math
from datetime import datetime,timezone,timedelta
import numpy as np
import rosbag,yaml
ROOT=Path('/home/unita/unita_ws');LOG=ROOT/'src/unita_waypoint/log';OUT=Path(__file__).resolve().parent
ZONE=timezone(timedelta(hours=9))
def wall(t):return datetime.fromtimestamp(t,ZONE).isoformat(timespec='milliseconds')
def read(path,topics):
 result={t:[] for t in topics}
 with rosbag.Bag(str(path)) as bag:
  start=bag.get_start_time();end=bag.get_end_time()
  for t,m,stamp in bag.read_messages(topics=topics):
   ts=stamp.to_sec()
   if hasattr(m,'data'):
    try:v=json.loads(m.data) if isinstance(m.data,str) and m.data.startswith('{') else m.data
    except ValueError:v=m.data
   elif t=='/imu':
    q=m.orientation;v={'yaw_deg':math.degrees(math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))),'yaw_rate_deg_s':math.degrees(m.angular_velocity.z)}
   else:v={k:float(getattr(m,k)) for k in ('steering','brake','accel','velocity')}
   result[t].append((ts,v))
 return start,end,result

def near(seq,t,max_gap=.10):
 if not seq:return None
 i=bisect.bisect_left([x[0] for x in seq],t)
 choices=seq[max(0,i-1):i+1];best=min(choices,key=lambda x:abs(x[0]-t))
 return best[1] if abs(best[0]-t)<=max_gap else None

def describe(values):
 v=np.asarray([x for x in values if x is not None and math.isfinite(x)],float)
 return {'min':float(v.min()),'max':float(v.max()),'std':float(v.std()),'mean':float(v.mean())} if len(v) else None

def metrics(rows):
 keys=('bearing_deg','wall_heading_deg','wall_clearance_m','steering_deg','imu_yaw_deg','imu_yaw_rate_deg_s','speed_kmh','target_speed_kmh')
 result={k:describe([r.get(k) for r in rows]) for k in keys};result['samples']=len(rows)
 if rows:result.update(start=wall(rows[0]['stamp']),end=wall(rows[-1]['stamp']),duration_s=rows[-1]['stamp']-rows[0]['stamp'])
 for key in ('bearing_deg','steering_deg'):
  signs=[math.copysign(1,r[key]) for r in rows if r.get(key) is not None and abs(r[key])>=.5]
  result[key+'_sign_reversals_ge_half_deg']=sum(a!=b for a,b in zip(signs,signs[1:]))
 return result

allresults={}
for run,hms in [('224300','22-43-00'),('224142','22-41-42'),('223335','22-33-35')]:
 semi=LOG/('semi_leo_2026-10-21-'+hms+'.bag');folder=LOG/('run_20261021_'+run)
 topics=['/leo/avoidance_status','/leo/driving_status','/leo/waypoint_debug/tracking_state','/leo/signal_status','/leo/raw_ctrl_cmd','/semi/raceline_ctrl_cmd','/ctrl_cmd','/competition/ego_speed_kmh']
 start,end,data=read(semi,topics);sensors={'/imu':[],'/control/udp_status':[]}
 for sensor in sorted(folder.glob('sensors_*.bag')):
  _,_,part=read(sensor,list(sensors))
  for topic,seq in part.items():sensors[topic].extend(seq)
 params=yaml.safe_load((folder/'rosparams.yaml').read_text())
 ftg=params['leo_driving_state'];follower=params['leo_waypoint_follower'];rows=[]
 for t,d in data['/leo/avoidance_status']:
  drive=near(data['/leo/driving_status'],t) or {};tracking=near(data['/leo/waypoint_debug/tracking_state'],t) or {}
  if drive.get('mode')!='BLACKOUT':continue
  cmd=near(data['/ctrl_cmd'],t) or {};raw=near(data['/leo/raw_ctrl_cmd'],t) or {};other=near(data['/semi/raceline_ctrl_cmd'],t) or {};imu=near(sensors['/imu'],t) or {};guard=near(data['/leo/signal_status'],t) or {}
  angle=d.get('target_bearing_rad',0);dist=ftg.get('target_distance_from_lidar_m',6);x=ftg.get('lidar_to_rear_x_m',1.4)+dist*math.cos(angle);y=dist*math.sin(angle)
  expected=math.atan2(2*follower['wheelbase']*y,x*x+y*y)*follower['steering_sign']
  rows.append({'stamp':t,'relative_s':round(t-start,4),'time':wall(t),'sector':tracking.get('sector'),'drive_mode':drive.get('mode'),'drive_valid':drive.get('valid'),'avoidance_valid':d.get('valid'),'tracking_stop_reason':tracking.get('stop_reason'),'phase':d.get('avoidance_phase'),'reason':d.get('reason'),'bearing_deg':math.degrees(angle),'wall_heading_deg':math.degrees(d['right_wall_heading_rad']) if d.get('right_wall_heading_rad') is not None else None,'wall_clearance_m':d.get('right_wall_clearance_m'),'wall_target_clearance_m':d.get('right_wall_target_clearance_m'),'wall_follow_weight':d.get('wall_follow_weight'),'remaining_rejoin_m':d.get('route_rejoin_remaining_m'),'front_points':d.get('front_points'),'front_clearance_m':d.get('front_clearance_m'),'nearest_front_x_m':d.get('nearest_front_x_m'),'obstacle_points':d.get('obstacle_points'),'crossing_threats':d.get('crossing_threats'),'tracked_obstacles':d.get('tracked_obstacles'),'candidate_bearings_deg':[math.degrees(x) for x in d.get('candidate_bearings_rad',[])],'speed_kmh':d.get('ego_speed_mps',0)*3.6 if d.get('ego_speed_mps') is not None else None,'target_speed_kmh':d.get('target_speed_mps',0)*3.6,'steering_deg':math.degrees(cmd['steering']) if 'steering' in cmd else None,'raw_steering_deg':math.degrees(raw['steering']) if 'steering' in raw else None,'raceline_steering_deg':math.degrees(other['steering']) if 'steering' in other else None,'raw_brake':raw.get('brake'),'raceline_brake':other.get('brake'),'brake':cmd.get('brake'),'expected_pp_steering_deg':math.degrees(expected),'imu_yaw_deg':imu.get('yaw_deg'),'imu_yaw_rate_deg_s':imu.get('yaw_rate_deg_s'),'drive_yaw_deg':math.degrees(drive['driving_pose'][2]) if drive.get('driving_pose') else None,'guard_state':guard.get('state'),'guard_reason':guard.get('reason'),'udp_status':near(sensors['/control/udp_status'],t,.15)})
 # Remove status repeats: plan identity excludes continually updated ego speed.
 unique=[]
 for r in rows:
  identity=tuple(r.get(k) for k in ('phase','bearing_deg','wall_clearance_m','wall_heading_deg','front_points','obstacle_points','target_speed_kmh'))
  if not unique or identity!=unique[-1][0]:unique.append((identity,r))
 plans=[r for identity,r in unique]
 segments=[]
 for r in rows:
  key=(r['sector'],r['phase'],r['wall_follow_weight']==1.)
  if not segments or segments[-1]['key']!=key:segments.append({'key':key,'rows':[]})
  segments[-1]['rows'].append(r)
 segments=[{'key':k['key'],'metrics':metrics(k['rows'])} for k in segments]
 jumps=[]
 for a,b in zip(plans,plans[1:]):
  if b['stamp']-a['stamp']>.5:continue
  jumps.append({'delta_bearing_deg':b['bearing_deg']-a['bearing_deg'],'delta_wall_heading_deg':b['wall_heading_deg']-a['wall_heading_deg'] if a['wall_heading_deg'] is not None and b['wall_heading_deg'] is not None else None,'before':a,'after':b})
 jumps.sort(key=lambda x:abs(x['delta_bearing_deg']),reverse=True)
 clean=[r for r in rows if r['sector']==14 and r['phase']=='WALL_FOLLOW' and r['wall_follow_weight']==1. and r['front_points']==0 and not r['crossing_threats']]
 result={'bag':str(semi),'start':wall(start),'end':wall(end),'duration_s':end-start,'source_head':'9bef9d0 before tunnel edit','relevant_ftg_parameters':ftg,'blackout_summary':metrics(rows),'s14_wall_follow_full_weight_no_front_threat':metrics(clean),'s14_passing':metrics([r for r in rows if r['sector']==14 and r['phase']=='PASSING']),'phase_segments':segments,'largest_target_jumps':jumps[:15],'rows':rows,'unique_plan_rows':plans,'limits':['Recorded target/status/control evidence only; nearest message matching (100 ms for commands/status, 150 ms for UDP).','Main raceline debug may report GPS stale in BLACKOUT; actual owner is Leo shared PP/PID with FTG target.','No live ROS nodes, vehicle commands, or source edits were used.']}
 out=OUT/(run+'_heading_evidence.json');out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 allresults[run]={k:result[k] for k in ('bag','blackout_summary','s14_wall_follow_full_weight_no_front_threat','s14_passing','phase_segments')}
 print(run,json.dumps(allresults[run],ensure_ascii=False),flush=True)
(OUT/'summary.json').write_text(json.dumps(allresults,indent=2,ensure_ascii=False)+'\n')
