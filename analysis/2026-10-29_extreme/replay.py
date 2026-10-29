"""Offline same-recorded-input sequential comparison; no ROS publishers."""
import sys,json
from pathlib import Path
from dataclasses import replace,asdict
from collections import Counter
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/2026-10-21_s14_heading'))
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
from replay_recorded_clouds import collect,guidance
from compare_heading import load
from leo_avoidance.ftg_local import Config,FollowTheGap
baseline=load(ROOT/'rollback/2026-10-29_extreme_launch_before/src/leo_avoidance/src/leo_avoidance/ftg_local.py','extreme_before')
values=yaml.safe_load((ROOT/'src/leo_avoidance/config/avoidance_local.yaml').read_text())
extra=yaml.safe_load((ROOT/'src/leo_avoidance/config/avoidance_extreme.yaml').read_text())
frames,route,points,params=collect('224300')
clock=[0.]
old=baseline.FollowTheGap(baseline.Config(**{k:v for k,v in values.items() if k in vars(baseline.Config())}),clock=lambda:clock[0])
new=FollowTheGap(Config(**{k:v for k,v in dict(values,**extra).items() if k in vars(Config())}),clock=lambda:clock[0])
legacy=FollowTheGap(Config(**{k:v for k,v in values.items() if k in vars(Config())}),clock=lambda:clock[0])
rows=[];last=None;legacy_diff=0
for i,f in enumerate(frames):
 clock[0]=f['stamp']-frames[0]['stamp']
 last,index,kw=guidance(f,route,last)
 kw['ground_reference_z_m']=values.get('s14_ground_reference_z_m')
 for p in (old,new,legacy):p.config=replace(p.config,max_speed_mps=points[index]['speed_kmh']/3.6)
 a=old.plan(f['points'],f['speed'],**kw);b=new.plan(f['points'],f['speed'],**kw);c=legacy.plan(f['points'],f['speed'],**kw)
 legacy_diff+=asdict(a)!=asdict(c)
 rows.append(dict(t=clock[0],actual_kmh=f['speed']*3.6,old=asdict(a),new=asdict(b)))
 if i%50==0:print('replayed',i,'/',len(frames),flush=True)
def stats(key):
 speeds=[r[key]['speed_mps']*3.6 for r in rows];phases=[r[key]['avoidance_phase'] for r in rows]
 durations=Counter()
 for a,b in zip(rows,rows[1:]):durations[a[key]['avoidance_phase']]+=min(.3,max(0,b['t']-a['t']))
 return dict(mean_target_kmh=float(np.mean(speeds)),max_target_kmh=max(speeds),over_30=sum(v>30+1e-8 for v in speeds),phase_scans=dict(Counter(phases)),phase_duration_s=dict(durations),passing_entries=sum(p=='PASSING' and (i==0 or phases[i-1]!='PASSING') for i,p in enumerate(phases)),wall_follow_mean_kmh=float(np.mean([v for v,p in zip(speeds,phases) if p=='WALL_FOLLOW'])))
report=dict(run='224300',samples=len(rows),old=stats('old'),new=stats('new'),legacy_differences=legacy_diff,recorded_actual_mean_kmh=float(np.mean([r['actual_kmh'] for r in rows])),limitations=['Reconstructed pose/guidance and matched clouds; not exact original worker inputs.','Both planners follow identical recorded vehicle motion, not a closed-loop simulation.','Current ground calibration used for both; these replay averages are not original recorded target averages.'],rows=rows)
Path(__file__).with_name('replay.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
