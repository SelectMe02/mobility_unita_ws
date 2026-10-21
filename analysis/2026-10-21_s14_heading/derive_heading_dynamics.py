"""Derived measured oscillation evidence; no replay or vehicle commands."""
from pathlib import Path
import json,math
import numpy as np
ROOT=Path(__file__).resolve().parent
result={}
for run in ('224300','224142','223335'):
 d=json.loads((ROOT/(run+'_heading_evidence.json')).read_text());rows=d['rows']
 straight=[r for r in rows if r['sector']==14 and r['phase']=='WALL_FOLLOW' and r['wall_follow_weight']==1 and r['wall_heading_deg'] is not None and r['imu_yaw_deg'] is not None]
 wall=np.array([r['wall_heading_deg'] for r in straight]);yaw=np.array([r['imu_yaw_deg'] for r in straight]);world=wall+yaw
 out={'straight_wall_follow_samples':len(straight),'wall_relative_heading_std_deg':float(wall.std()),'wall_relative_heading_range_deg':float(np.ptp(wall)),'imu_yaw_std_deg':float(yaw.std()),'imu_yaw_range_deg':float(np.ptp(yaw)),'world_wall_heading_std_deg':float(world.std()),'world_wall_heading_range_deg':float(np.ptp(world)),'wall_heading_vs_imu_yaw_correlation':float(np.corrcoef(wall,yaw)[0,1])}
 # Reconstruct only the documented .65/.35 clearance update from distinct plans.
 # This is a preference diagnostic, not an exact gap/collision replay.
 filtered=None;opposite=[]
 for r in d['unique_plan_rows']:
  if r['phase']=='ROUTE':filtered=None
  c=r['wall_clearance_m'];heading=r['wall_heading_deg']
  if c is not None:filtered=c if filtered is None else .65*filtered+.35*c
  if r['phase']!='WALL_FOLLOW' or r['wall_follow_weight']!=1 or r['sector']!=14 or filtered is None or heading is None:continue
  error=r['wall_target_clearance_m']-filtered;error=math.copysign(max(0,abs(error)-.1),error)
  preferred=float(np.clip(math.degrees(.25*error)+heading,-math.degrees(.22),math.degrees(.22)))
  if preferred*r['bearing_deg']<0 and abs(preferred)>.7 and abs(r['bearing_deg'])>.7:
   opposite.append({'time':r['time'],'relative_s':r['relative_s'],'wall_clearance_m':c,'reconstructed_filtered_clearance_m':filtered,'wall_heading_deg':heading,'reconstructed_unfiltered_preference_deg':preferred,'selected_target_deg':r['bearing_deg'],'recorded_steering_deg':r['steering_deg'],'imu_yaw_rate_deg_s':r['imu_yaw_rate_deg_s'],'front_points':r['front_points'],'crossing_threats':r['crossing_threats']})
 out['opposite_preference_examples']=opposite
 end=max(r['stamp'] for r in straight);preexit=[r for r in rows if r['sector']==14 and r['stamp']<=end and r['imu_yaw_deg'] is not None]
 peaks=[];valleys=[]
 for i,r in enumerate(preexit):
  if r['stamp']-preexit[0]['stamp']<1 or preexit[-1]['stamp']-r['stamp']<1:continue
  neighborhood=[x['imu_yaw_deg'] for x in preexit if abs(x['stamp']-r['stamp'])<=1.]
  for outlist,cmp in ((peaks,max),(valleys,min)):
   if r['imu_yaw_deg']==cmp(neighborhood) and (not outlist or r['stamp']-outlist[-1]['stamp']>1.):outlist.append({k:r[k] for k in ('stamp','time','phase','imu_yaw_deg','bearing_deg','wall_clearance_m')})
 out['preexit_imu_yaw_maxima']=peaks;out['preexit_imu_yaw_minima']=valleys
 out['maxima_periods_s']=[b['stamp']-a['stamp'] for a,b in zip(peaks,peaks[1:])]
 out['minima_periods_s']=[b['stamp']-a['stamp'] for a,b in zip(valleys,valleys[1:])]
 out['limits']=['World wall heading uses current recorded IMU and latest published plan; scan latency is not compensated.','Preference reconstruction uses distinct published plans, .65/.35 clearance update, and recorded gains; gap clipping, safety checks, and raw clouds are not replayed.','Oscillation periods include PASSING intervals and characterize the actual recorded motion; they are not exclusively obstacle-free closed-loop periods.']
 result[run]=out
(ROOT/'heading_dynamics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
