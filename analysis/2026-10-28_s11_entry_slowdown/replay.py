"""Recorded ego poses with an explicitly INJECTED R1 NPC at the requested point."""
import json,sys,math,importlib.util
from pathlib import Path
import rosbag
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
import local_avoidance_core as c
spec=importlib.util.spec_from_file_location('helper',ROOT/'analysis/2026-10-22_highway_side_vehicle/replay_lane_change.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
run=ROOT/'src/unita_waypoint/log/run_20261028_180737'
p,route,sectors,speeds=h.build(c,run/'config/map_tunner_project.json')
rows=[]
for bp in run.glob('*.bag'):
 with rosbag.Bag(str(bp)) as b:
  for _,m,t in b.read_messages(['/waypoint_debug/control_decision']):
   d=json.loads(m.data);i=d.get('waypoint_index')
   if i is None or sectors[i]!=11 or d.get('x') is None:continue
   ego=(d['x'],d['y']);a,bp=route[i],route[min(i+1,len(route)-1)]
   yaw=math.atan2(bp[1]-a[1],bp[0]-a[0])
   npc=c.Obstacle(*c.ENTRY,2.,length=4.5,width=1.9,yaw=math.atan2(c.TANGENT[1],c.TANGENT[0]),confirmed=False,velocity_reliable=False)
   forward=(npc.x-ego[0])*math.cos(yaw)+(npc.y-ego[1])*math.sin(yaw)
   obstacles=[npc] if c.s11_entry_monitor_active(ego) and 0.<forward<c.S11_ENTRY_FRONT_RANGE_M else []
   out=p.plan(i,ego,(d['current_speed_kmh'] or 0.)/3.6,obstacles,ego_yaw=yaw,desired_speed_mps=speeds[i]/3.6,observation_stamp=d['stamp'])
   if p.s11_entry_guard.active:
    assert out.speed_cap_kmh<=30.
    assert p.transition is None
    rows.append(dict(x=ego[0],y=ego[1],recorded_speed=d['current_speed_kmh'],target=out.speed_cap_kmh,reason=out.reason))
report=dict(input='Recorded ego poses, injected NPC at (64,125.85); NOT proof of recorded NPC detection or closed-loop collision avoidance.',frames=len(rows),first=rows[0] if rows else None,last=rows[-1] if rows else None,rows=rows)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(report,indent=2))
print({k:v for k,v in report.items() if k!='rows'})
