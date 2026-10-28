"""Real recorded obstacles/ego inputs; no injected NPC or vehicle simulation."""
import sys,json,importlib.util
from pathlib import Path
import rosbag
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/unita_waypoint/scripts'))
import local_avoidance_core as new
from pedal_guard import CruiseBrake
spec=importlib.util.spec_from_file_location('entry_old',ROOT/'rollback/2026-10-28_s11_entry_brake85_before/src/unita_waypoint/scripts/local_avoidance_core.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
report=[]
for run in sorted((ROOT/'src/unita_waypoint/log').glob('run_20261028_182*'))[-4:]:
 guards=[old.EntryGuard(),new.EntryGuard()];first=[None,None];rows=[]
 brake=CruiseBrake(gain=.08,max_brake=.85,rise_rate=3.)
 last=None
 for bp in run.glob('*.bag'):
  with rosbag.Bag(str(bp)) as bag:
   start=bag.get_start_time()
   for _,m,ts in bag.read_messages(['/waypoint_debug/control_decision']):
    d=json.loads(m.data)
    if d.get('x') is None:continue
    ego=(d['x'],d['y']);stamp=ts.to_sec();dt=.05 if last is None else stamp-last;last=stamp
    objects=[new.Obstacle(**{k:v for k,v in o.items() if k in new.Obstacle.__dataclass_fields__}) for o in d['obstacles']]
    on=[]
    for k,g in enumerate(guards):
     on.append(g.update(11,d.get('current_lane',1),ego,objects))
     if on[-1] and first[k] is None:
      first[k]=dict(t=round(stamp-start,3),x=ego[0],y=ego[1],speed=d['current_speed_kmh'],recorded_brake=d['command']['brake'])
    if on[1]:
     out=brake.shape(-1.,d['current_speed_kmh'],30.,dt)
     rows.append(dict(t=round(stamp-start,3),new_pedal_at_recorded_speed=-out,recorded_brake=d['command']['brake'],speed=d['current_speed_kmh']))
    else:brake.reset()
 report.append(dict(run=run.name,first_old=first[0],first_new=first[1],frames=len(rows),rows=rows))
 print({k:v for k,v in report[-1].items() if k!='rows'},flush=True)
(Path(__file__).parent/'evidence.json').write_text(json.dumps(dict(limitation='Recorded-input trigger and pedal comparison; no prediction of actual vehicle motion or collision outcome.',runs=report),indent=2))
