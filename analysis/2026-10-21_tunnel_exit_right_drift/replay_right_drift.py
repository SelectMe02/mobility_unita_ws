"""Compare exit guidance on the recorded drive and original LiDAR clouds.

No publishers or ROS nodes. Fixed-state comparisons isolate the changed
guidance; they do not predict a new vehicle trajectory.
"""
import copy
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import rosbag
import yaml
from leo_avoidance.ftg_local import Config, FollowTheGap
from leo_guard.route_file import anchor_on_route, load_route_points
from leo_guard.traffic_local import RouteProgress

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis/2026-10-21_s14_heading'))
from replay_recorded_clouds import collect


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def setup(module,params,project):
    state=module.DrivingStateNode.__new__(module.DrivingStateNode)
    state.route=RouteProgress(load_route_points(str(project)),max_error=7.,backward_window=30)
    for name in ('route_lookahead_m','exit_rejoin_start_m','exit_rejoin_full_m'):
        setattr(state,name,float(params[name]))
    state.exit_route_s=anchor_on_route(state.route.points,params['exit_anchor'])[1]
    state.exit_route_offset=np.zeros(2)
    state.exit_rejoin_active=False
    state.s14_ground_bounds=module.s14_ground_interval(str(project),state.route)
    state.s14_ground_reference_z_m=float(params.get('s14_ground_reference_z_m',-1.61))
    return state


if __name__=='__main__':
    relative=Path('src/leo_avoidance/scripts/driving_state_node_local.py')
    old=load(ROOT/'rollback/2026-10-21_tunnel_exit_right_drift_before'/relative,'right_drift_before')
    new=load(ROOT/relative,'right_drift_after')
    directory=ROOT/'src/unita_waypoint/log/run_20261021_234918'
    project=directory/'config/map_tunner_project.json'
    frames,_,points,params=collect('234918')
    before_node,after_node=(setup(m,params,project) for m in (old,new))
    config=Config(**{k:params.get(k,v) for k,v in vars(Config()).items()})
    clock=[0.]
    before=FollowTheGap(config,clock=lambda:clock[0])
    rows=[]
    for frame in frames:
        clock[0]=frame['stamp']
        preferred=before_node.route_bearing(frame['pose'])
        after_node.route_bearing(frame['pose'])
        progress=before_node.route_progress_m
        remaining=frame['recorded_plan'].get('route_rejoin_remaining_m')
        inputs=[]
        for state in (before_node,after_node):
            if remaining is None:
                heading,weight,rem=None,1.,None
            else:
                state.route_progress_m=state.exit_route_s-remaining
                state.exit_rejoin_active=True
                heading,weight,rem=state.exit_rejoin_guidance(frame['pose'][2])
            inputs.append(dict(preferred_bearing_rad=preferred,wall_follow=True,
                               wall_follow_weight=weight,route_rejoin_remaining_m=rem,
                               exit_heading_error_rad=heading,
                               ego_pose=frame['pose'],stamp=frame['stamp'],
                               ground_reference_z_m=state.ground_reference_for_progress(progress)))
        after=FollowTheGap(config,clock=lambda:clock[0])
        after.__dict__.update(copy.deepcopy(before.__dict__))
        before.config=replace(config,max_speed_mps=points[before_node.route.index]['speed_kmh']/3.6)
        after.config=before.config
        a=before.plan(frame['points'],frame['speed'],**inputs[0])
        b=after.plan(frame['points'],frame['speed'],**inputs[1])
        av,bv=asdict(a),asdict(b)
        changed=[k for k in av if av[k]!=bv[k]]
        protected=remaining is None or remaining>=5.
        assert not protected or not changed
        # The selected positive-speed target must still pass physical collision checks.
        selected=after._remove_ego_returns(after._in_collision_height(
            after._ground_filter(frame['points'],inputs[1]['ground_reference_z_m'])[0],
            inputs[1]['ground_reference_z_m'],
            after._ground_filter(frame['points'],inputs[1]['ground_reference_z_m'])[1]))
        if b.speed_mps>0.:
            assert after._swept_body_collision(selected,b.target_bearing_rad) is None
        rows.append(dict(stamp=frame['stamp'],match_error_s=frame['match_error_s'],
                         remaining_m=remaining,protected=protected,changed_fields=changed,
                         before_guidance_deg=None if inputs[0]['exit_heading_error_rad'] is None else math.degrees(inputs[0]['exit_heading_error_rad']),
                         after_guidance_deg=None if inputs[1]['exit_heading_error_rad'] is None else math.degrees(inputs[1]['exit_heading_error_rad']),
                         before=av,after=bv))
    result=dict(run='234918',compared=len(rows),
                unchanged_approach_scans=sum(r['protected'] for r in rows),
                changed_exit_scans=sum(bool(r['changed_fields']) for r in rows),
                match_error_p95_s=float(np.percentile([r['match_error_s'] for r in rows],95)),
                limitations=['Fixed-state original-cloud comparison; exact worker odometry and clock inputs were not recorded.',
                             'The changed target has not been validated in a new vehicle drive.'],samples=rows)
    (OUT/'cloud_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))
