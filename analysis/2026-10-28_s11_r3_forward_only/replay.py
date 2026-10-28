"""Recorded-input replay only: no ROS publishers, no counterfactual vehicle motion."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from collections import Counter
import rosbag
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src/unita_waypoint/scripts'))

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

helper = module(ROOT/'analysis/2026-10-22_highway_side_vehicle/replay_lane_change.py', 'replay_helper')
old = module(ROOT/'rollback/2026-10-28_s11_r3_forward_only_before/src/unita_waypoint/scripts/local_avoidance_core.py', 's11_before')
new = module(ROOT/'src/unita_waypoint/scripts/local_avoidance_core.py', 's11_after')
from avoidance_runtime import SETTINGS

report = {'limitation': 'Open-loop recorded-input replay; ego keeps its recorded pose/speed. Does not prove collision-free execution.', 'runs': []}
for run in sorted((ROOT/'src/unita_waypoint/log').glob('run_20261028_165*'))[-3:]:
    if not (run/'rosparams.yaml').exists() or not list(run.glob('*.bag')):
        continue
    params = yaml.safe_load((run/'rosparams.yaml').read_text())['gps_waypoint_follower']
    planners = []
    for core in (old, new):
        p, route, sectors, speeds = helper.build(core, run/'config/map_tunner_project.json')
        for name, spec in SETTINGS.items():
            if spec[0] == 'gps_waypoint_follower' and name in params:
                setattr(p, spec[2], params[name])
        planners.append(p)
    result = {'run': run.name, 'frames': 0, 'recorded_events': [], 'replay_events': [],
              'original_approval_comparison': [], 'rear_evidence': []}
    counts = [Counter(), Counter()]
    starts = [0,0]
    cancellations = [0,0]
    previous = [None, None]
    recorded_previous = None
    obstacle_stamp = None
    for bag_path in sorted(run.glob('*.bag')):
        with rosbag.Bag(str(bag_path)) as bag:
            t0 = bag.get_start_time()
            for topic, message, ts in bag.read_messages(['/waypoint_debug/control_decision','/waypoint_debug/obstacles']):
                data = json.loads(message.data)
                if topic.endswith('/obstacles'):
                    obstacle_stamp = data['stamp']
                    continue
                index = data.get('waypoint_index')
                if index is None or not 0 <= index < len(sectors) or sectors[index] != 11 or data.get('x') is None:
                    continue
                result['frames'] += 1
                t = round(ts.to_sec()-t0, 3)
                a,b = route[index],route[min(index+1,len(route)-1)]
                yaw = math.atan2(b[1]-a[1],b[0]-a[0])
                tangent = (math.cos(yaw),math.sin(yaw))
                recorded_tr = data.get('transition')
                recorded_key = (data.get('state'), recorded_tr and (recorded_tr['source'],recorded_tr['target']))
                if recorded_key != recorded_previous:
                    result['recorded_events'].append(dict(t=t, state=data.get('state'), reason=data.get('lane_change_reason'), transition=recorded_tr))
                recorded_previous = recorded_key
                outputs = []
                for k,(core,p) in enumerate(zip((old,new), planners)):
                    obstacles=[]
                    for raw in data['obstacles']:
                        fields = {name:raw[name] for name in core.Obstacle.__dataclass_fields__ if name in raw}
                        fields['track_id'] = raw.get('id',0)
                        obstacles.append(core.Obstacle(**fields))
                    before = dict(p.transition) if p.transition else None
                    dec = p.plan(index,(data['x'],data['y']),(data['current_speed_kmh'] or 0.)/3.6,
                                 obstacles,ego_yaw=yaw,desired_speed_mps=speeds[index]/3.6,
                                 observation_stamp=obstacle_stamp)
                    tr = p.transition
                    if k == 1 and tr is not None:
                        assert tr['target'] == tr['source']+1, (run.name, t, tr)
                        assert tr['state'] != 'CANCELLING'
                    counts[k][dec.reason or dec.state] += 1
                    started = tr is not None and before is None
                    starts[k] += int(started)
                    cancelled = tr is not None and tr['state']=='CANCELLING' and (before is None or before['state']!='CANCELLING')
                    cancellations[k] += int(cancelled)
                    key = (dec.state,dec.reason,tr and (tr['source'],tr['target']))
                    if previous[k] != key:
                        result['replay_events'].append(dict(t=t,version=('before','after')[k],state=dec.state,reason=dec.reason,
                                                           target=tr and tr['target'], index=index))
                    previous[k]=key
                    outputs.append(dict(state=dec.state,reason=dec.reason,target=tr and tr['target']))
                if data.get('lane_change_reason')=='mobil_gain':
                    result['original_approval_comparison'].append(dict(t=t,index=index,before=outputs[0],after=outputs[1]))
                    for raw in data['obstacles']:
                        dx,dy=raw['x']-data['x'],raw['y']-data['y']
                        along=dx*tangent[0]+dy*tangent[1]
                        lateral=-dx*tangent[1]+dy*tangent[0]
                        if along<0:
                            result['rear_evidence'].append(dict(t=t,id=raw['id'],along=round(along,2),lateral=round(lateral,2),
                                lane=raw['raceline'],confirmed=raw['confirmed'],velocity_reliable=raw['velocity_reliable'],
                                vx=raw['vx'],vy=raw['vy'],target=recorded_tr['target']))
    result.update(before_starts=starts[0],after_starts=starts[1],before_cancels=cancellations[0],after_cancels=cancellations[1],
                  reason_counts={'before':dict(counts[0]),'after':dict(counts[1])})
    report['runs'].append(result)
    print(run.name, result['frames'], 'starts',starts,'cancels',cancellations,flush=True)
    print('Original approvals:',result['original_approval_comparison'],flush=True)
(OUT/'evidence.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
