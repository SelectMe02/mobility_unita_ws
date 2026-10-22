"""Read-only GPS exit decision replay. No ROS node, publisher or vehicle run."""
import bisect
from collections import deque
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys

import rosbag
import yaml
from leo_avoidance.driving_state_local import DrivingState
from leo_avoidance.scan_odometry import ScanOdometry
from leo_guard.route_file import anchor_on_route, load_route_points
from leo_guard.traffic_local import RouteProgress

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LOG = ROOT / 'src/unita_waypoint/log'
NODE = ROOT / 'src/leo_avoidance/scripts/driving_state_node_local.py'
spec = importlib.util.spec_from_file_location('exit_replay_node', NODE)
node = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = node
spec.loader.exec_module(node)
ZONE = timezone(timedelta(hours=9))


def wall(stamp):
    return datetime.fromtimestamp(stamp, ZONE).isoformat(timespec='milliseconds')


def run(folder, bagfile):
    directory = LOG / folder
    config = yaml.safe_load((directory / 'rosparams.yaml').read_text())['leo_driving_state']
    project = directory / 'config/map_tunner_project.json'
    points = load_route_points(str(project))
    poses, speeds, statuses, exit_scans = [], [], [], []
    for bagpath in directory.glob('sensors_*.bag'):
        with rosbag.Bag(str(bagpath)) as bag:
            for topic, message, stamp in bag.read_messages(topics=[
                    '/localization/pose', '/competition/ego_speed_kmh']):
                t = stamp.to_sec()
                if topic.endswith('/pose'):
                    p, q = message.pose.position, message.pose.orientation
                    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
                    poses.append(((p.x, p.y, yaw), t, message.header.stamp.to_sec()))
                else:
                    speeds.append((t, message.data / 3.6))
    poses.sort(key=lambda p:p[1]); speeds.sort()
    with rosbag.Bag(str(LOG / bagfile)) as bag:
        start = bag.get_start_time()
        for topic, message, stamp in bag.read_messages(topics=[
                '/leo/driving_status','/leo/avoidance_status','/ctrl_cmd']):
            if topic.endswith('driving_status'):
                statuses.append((stamp.to_sec(),json.loads(message.data)))
            elif topic.endswith('avoidance_status'):
                plan=json.loads(message.data)
                if plan.get('valid') and plan.get('route_rejoin_remaining_m') is not None:
                    exit_scans.append(stamp.to_sec())
    times = [p[1] for p in poses]; speedtimes = [s[0] for s in speeds]
    rows=[]
    for stamp, recorded in statuses:
        if not recorded.get('gps_return_elapsed_s') or recorded.get('raw_gps_kind')!='valid':
            continue
        index=bisect.bisect_right(times,stamp)-1
        si=bisect.bisect_right(speedtimes,stamp)-1
        if index<0 or si<0: continue
        state=node.DrivingStateNode.__new__(node.DrivingStateNode)
        state.machine=DrivingState(ScanOdometry(),input_timeout_s=config['input_timeout_s'],
                                   max_blackout_distance_m=config['max_blackout_distance_m'])
        machine=state.machine
        machine.had_blackout=True
        machine.map_pose=recorded['map_pose']
        machine.gps_return_since=stamp-recorded['gps_return_elapsed_s']
        machine.raw_gps=('valid',stamp)
        machine.gps_valid=(True,stamp-(recorded.get('gps_valid_age_s') or 0.))
        machine.gps_pose=(poses[index][0],poses[index][1])
        machine.scan_valid=recorded.get('scan_fault_reason') is None
        machine.last_scan_wall=stamp-(recorded.get('scan_age_s') or 0.)
        machine.update_odom(tuple(recorded['map_pose']),stamp)
        machine.update_avoidance(True,stamp)
        state.route=RouteProgress(points,max_error=7.)
        state.blackout_route=RouteProgress(points,max_error=7.,backward_window=30)
        state.rejoin_route=RouteProgress(points,max_error=2.)
        _,state.exit_route_s=anchor_on_route(points,config['exit_anchor'])
        # Original published exit remaining-distance proves the exit latch was entered.
        state.exit_rejoin_active=any(t<stamp for t in exit_scans)
        state.gps_history=deque(poses[max(0,index-30):index+1],maxlen=100)
        state.latest_speed=(speeds[si][1],speeds[si][0]);state.speed_timeout_s=.6
        state.rejoin_diagnostic=None
        old=state.route_rejoin_ok(machine.gps_pose[0],machine.map_pose,
                                  recorded.get('blackout_route_index'))
        olddiag=state.rejoin_diagnostic
        new=old or state.exit_gps_rejoin_ok(stamp)
        diagnostic=state.rejoin_diagnostic
        mode,pose,reason=machine.select(stamp,recorded.get('blackout_allowed',False),new)
        rows.append(dict(stamp=stamp,time=wall(stamp),relative_s=stamp-start,
                         recorded_mode=recorded['mode'],recorded_reason=recorded['reason'],
                         old_rejoin=old,new_rejoin=new,proposed_mode=mode,
                         old_diagnostic=olddiag,new_diagnostic=diagnostic,
                         gps_lidar_error_m=recorded.get('gps_rejoin_error'),
                         proposed_reason=reason))
    accepted=next((r for r in rows if r['new_rejoin']),None)
    stopped=next((dict(time=wall(t),stamp=t,relative_s=t-start,**d) for t,d in statuses
                  if t-start>40 and d['mode']=='HOLD'),None)
    return dict(bag=bagfile,first_new_acceptance=accepted,first_recorded_exit_hold=stopped,
                rows=rows)


if __name__=='__main__':
    result=dict(limitations=[
        'Replays the return decision on original GPS and recorded pre-decision ICP poses.',
        'Bag receive times approximate monotonic callback receipt times.',
        'Does not predict the changed vehicle trajectory after GPS rejoin.',
        'Existing recorded blackout entry/return and validity are used; no simulator truth.'],
        runs=[run('run_20261021_231857','semi_leo_2026-10-21-23-18-56.bag')])
    (OUT/'replay_result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for r in result['runs']:
        print(json.dumps({k:v for k,v in r.items() if k!='rows'},ensure_ascii=False,indent=2))
