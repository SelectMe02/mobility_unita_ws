#!/usr/bin/env python3
"""Camera signal gate for PP/PID; lane detection is not a drive prerequisite."""
import copy
import json
import math
import sys
import threading
import time
from pathlib import Path
import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped
from morai_msgs.msg import CtrlCmd
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import SetBool, SetBoolResponse
from leo_guard.traffic_local import RouteProgress, TrafficDecision
from leo_guard.route_file import (anchor_on_route, checkpoint_indices,
                                  load_lane_points, load_route_points, load_survey)
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from avoidance_runtime import RuntimeSettings  # noqa: E402


# The Raceline 1 projection may lie this far from the lane actually driven
# (Raceline 3 is two lanes over); the corridor check itself is lane-based.
LANE_REACH_M = 12.


def checkpoint_ranges(text, checkpoints):
    """'8-11,13-14' -> [(index of CP8, index of CP11), ...] on the Leo route."""
    ranges = []
    for item in filter(None, (part.strip() for part in str(text).split(','))):
        start, end = (part.strip() for part in item.split('-'))
        if start not in checkpoints or end not in checkpoints:
            raise ValueError('unknown checkpoint range {}'.format(item))
        ranges.append((checkpoints[start], checkpoints[end]))
    return ranges


def in_ranges(index, ranges):
    return index is not None and any(lo <= index <= hi for lo, hi in ranges)


class SignalGuard:
    def __init__(self):
        self.lock=threading.RLock();self.inputs={};self.armed=False
        self.pose_fault_since=None
        self.pose_fault_grace_s=float(rospy.get_param('~pose_fault_grace_s',0.0))
        if not math.isfinite(self.pose_fault_grace_s) or self.pose_fault_grace_s<0:
            raise ValueError('invalid pose fault grace')
        waypoint_file=rospy.get_param('~waypoint_file')
        points=load_route_points(waypoint_file)
        # Surveyed stop lines are map anchors projected onto the current route.
        stops=load_survey(rospy.get_param('~intersection_file'),points)
        stoplines=json.loads(Path(rospy.get_param('~stoplines_file')).read_text())
        # Corridor half-width: route_max_error_m outside the tunnel and
        # blackout_route_max_error_m inside the GPS-denied window (live from the panel).
        self.route_max_error_m=float(rospy.get_param('~route_max_error_m',3.75))
        self.blackout_route_max_error_m=float(rospy.get_param(
            '~blackout_route_max_error_m',self.route_max_error_m))
        self.blackout=[]
        if rospy.has_param('~blackout/blackout_start_anchor'):
            self.blackout=[(anchor_on_route(points,rospy.get_param('~blackout/blackout_start_anchor'))[0],
                            anchor_on_route(points,rospy.get_param('~blackout/blackout_end_anchor'))[0])]
        # Pass-through checkpoint segments: another follower's command is
        # sent unchanged (no signal gate, no route corridor fault).
        checkpoints=checkpoint_indices(waypoint_file) if waypoint_file.lower().endswith('.json') else {}
        self.passthrough=checkpoint_ranges(rospy.get_param('~passthrough_checkpoints',''),checkpoints)
        self.passthrough_max_error_m=float(rospy.get_param('~passthrough_route_max_error_m',15.))
        self.in_passthrough=False
        self.last_exit_rejoin_stamp=None
        self.route=RouteProgress(points,max_error=self.route_max_error_m,backward_window=30)
        # Lanes the followers may legitimately drive: saved Raceline 2/3
        # (local avoidance) and the route avoidance offset from Raceline 1.
        lanes=load_lane_points(waypoint_file)
        self.lanes=cKDTree(lanes) if lanes is not None else None
        # GPS sensor transform (0,0,1.2) was inspected in the simulator.
        # MORAI sensor origin is the rear axle; competition wheelbase is 3 m.
        self.front_axle=float(rospy.get_param('~gps_to_front_axle_m',0.))
        self.half_width=float(rospy.get_param('~vehicle_half_width_m',.946))
        if not all(math.isfinite(v) and 0<=v<=5. for v in (self.front_axle,self.half_width)):
            raise ValueError('invalid measured vehicle/GPS geometry')
        self.decision=TrafficDecision(stops,self.route.total,
            front_margin=float(rospy.get_param('~stop_margin_m',5.)),
            stoplines=stoplines,
            missed_signal_hold_s=float(rospy.get_param('~missed_signal_hold_s',0.)))
        # Arm by itself once every input has been fresh for auto_arm_hold_s.
        self.auto_arm=bool(rospy.get_param('~auto_arm',False))
        self.auto_arm_hold_s=float(rospy.get_param('~auto_arm_hold_s',1.))
        self.ready_since=None
        self.runtime=RuntimeSettings('leo_signal_guard',self)
        self.pub=rospy.Publisher('/ctrl_cmd',CtrlCmd,queue_size=1)
        self.cap=rospy.Publisher('/leo/speed_limit_kmh',Float64,queue_size=1)
        self.status=rospy.Publisher('/leo/signal_status',String,queue_size=1)
        self.camera=rospy.Publisher('/leo/signal_camera',String,queue_size=1,latch=True)
        self.subs=[rospy.Subscriber('/leo/raw_ctrl_cmd',CtrlCmd,lambda m:self.put('command',m),queue_size=1),
            rospy.Subscriber('/localization/pose',PoseStamped,self.pose,queue_size=1),
            rospy.Subscriber('/localization/valid',Bool,lambda m:self.put('valid',m.data),queue_size=1),
            rospy.Subscriber('/competition/ego_speed_kmh',Float64,lambda m:self.put('speed',m.data/3.6),queue_size=1),
            rospy.Subscriber('/perception/traffic_lights',String,self.signals,queue_size=1),
            rospy.Subscriber('/leo/driving_status',String,self.driving_status,queue_size=1)]
        route_avoidance_topic=rospy.get_param('~route_avoidance_topic','')
        if route_avoidance_topic:
            self.subs.append(rospy.Subscriber(route_avoidance_topic,String,self.route_offset,queue_size=1))
        if self.passthrough:
            self.subs.append(rospy.Subscriber(rospy.get_param('~passthrough_command_topic'),CtrlCmd,
                lambda m:self.put('passthrough_command',m),queue_size=1))
        self.service=rospy.Service('~set_armed',SetBool,self.set_armed)
        self.timer=rospy.Timer(rospy.Duration(.05),self.tick)
        rospy.logwarn('Signal guard: %s; pass-through route index ranges %s; blackout %s',
                      'arms automatically' if self.auto_arm else 'starts with brake held',
                      self.passthrough,self.blackout)

    def put(self,key,value):
        with self.lock:self.inputs[key]=(value,time.monotonic())

    def pose(self,m):
        age=rospy.Time.now().to_sec()-m.header.stamp.to_sec()
        q=m.pose.orientation
        value=(m.pose.position.x,m.pose.position.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        if m.header.frame_id!='map' or not 0<=age<=.35 or not all(math.isfinite(v) for v in value):value=None
        self.put('pose',value)

    def signals(self,m):
        value=None
        try:
            candidate=json.loads(m.data)
            age=rospy.Time.now().to_sec()-float(candidate['stamp'])
            if math.isfinite(age) and 0<=age<=1.:
                candidate['deadline']=time.monotonic()+1.-age;value=candidate
        except (ValueError,KeyError,TypeError):pass
        self.put('signals',value)

    def route_offset(self,m):
        try:offset=float(json.loads(m.data).get('offset_m',0.))
        except (ValueError,TypeError,AttributeError):return
        if math.isfinite(offset):self.put('route_offset',offset)

    def driving_status(self,m):
        """Capture a validated exit GPS handoff for one atomic route reset."""
        try:
            value=json.loads(m.data);diagnostic=value.get('gps_rejoin_diagnostic') or {}
            stamp=float(value['gps_rejoin_stamp']);pose=tuple(value['driving_pose'])
            if (value.get('mode')!='GPS' or not value.get('valid')
                    or diagnostic.get('method')!='continuous_exit_gps'
                    or diagnostic.get('route_ok') is not True
                    or len(pose)!=3 or not math.isfinite(stamp)
                    or not all(math.isfinite(v) for v in pose)):
                return
        except (ValueError,KeyError,TypeError,AttributeError):return
        self.put('exit_gps_rejoin',(stamp,pose))

    def consume_exit_gps_rejoin(self,pose,now):
        event=self.fresh('exit_gps_rejoin',now,.5)
        if event is None or event[0]==self.last_exit_rejoin_stamp:return False
        stamp,selected_pose=event
        if math.hypot(pose[0]-selected_pose[0],pose[1]-selected_pose[1])>2.:return False
        self.route.index=None;self.decision.reset();self.in_passthrough=False
        self.pose_fault_since=None;self.last_exit_rejoin_stamp=stamp
        return True

    def fresh(self,key,now,timeout=.35):
        item=self.inputs.get(key)
        return item[0] if item and 0<=now-item[1]<=timeout else None

    def corridor(self):
        """Route corridor half-width for the current route index."""
        index=self.route.index
        if in_ranges(index,self.passthrough):return max(self.passthrough_max_error_m,self.route_max_error_m)
        if in_ranges(index,self.blackout):return self.blackout_route_max_error_m
        return self.route_max_error_m

    def track(self,pose,now):
        """Route progress and the distance to the lane actually being driven."""
        self.route.max_error=self.corridor()+LANE_REACH_M
        progress,error=self.route.locate(*pose)
        corridor=self.corridor()
        lane_error=error
        offset=self.fresh('route_offset',now,.5)
        if offset:
            start=self.route.points[self.route.index]
            tangent=self.route.delta[self.route.index]/self.route.lengths[self.route.index]
            left=float(tangent[0]*(pose[1]-start[1])-tangent[1]*(pose[0]-start[0]))
            lane_error=min(lane_error,abs(left-offset))
        if self.lanes is not None and lane_error>corridor/2.:
            lane_error=min(lane_error,float(self.lanes.query(pose[:2])[0]))
        if lane_error>corridor:
            raise ValueError('vehicle is outside route corridor or facing backwards')
        return progress,lane_error

    def arm(self,now):
        pose=self.fresh('pose',now)
        missing=[]
        if self.fresh('valid',now) is not True:missing.append('driving_valid')
        if pose is None:missing.append('driving_pose')
        if self.fresh('speed',now) is None:missing.append('ego_speed')
        if missing:
            return False,'fresh input required: '+', '.join(missing)
        try:
            self.route.index=None;self.route.max_error=self.passthrough_max_error_m+LANE_REACH_M
            self.route.locate(*pose)
            self.track(pose,now)
        except ValueError as exc:return False,str(exc)
        self.armed=True;self.decision.reset();self.pose_fault_since=None
        self.in_passthrough=False
        return True,'armed'

    def set_armed(self,req):
        with self.lock:
            now=time.monotonic()
            if req.data:
                ok,message=self.arm(now)
                return SetBoolResponse(ok,message)
            self.armed=False;self.decision.reset()
            self.pose_fault_since=None;self.ready_since=None
        return SetBoolResponse(True,'brake held')

    def try_auto_arm(self,now):
        if not self.auto_arm or self.armed:
            return
        ready=(self.fresh('valid',now) is True and self.fresh('pose',now) is not None
               and self.fresh('speed',now) is not None)
        if not ready:
            self.ready_since=None;return
        if self.ready_since is None:self.ready_since=now
        if now-self.ready_since>=self.auto_arm_hold_s:
            ok,message=self.arm(now)
            self.ready_since=None
            if ok:rospy.loginfo('Signal guard armed automatically')
            else:rospy.logwarn_throttle(2.,'Signal guard auto-arm waiting: %s',message)

    def tick(self,event):
        output=CtrlCmd();output.longlCmdType=1;output.brake=1.;cap=0.
        with self.lock:
            self.runtime.poll()
            now=time.monotonic()
            self.try_auto_arm(now)
            detail=dict(state='hold',reason='automatic start waiting for inputs' if self.auto_arm
                        else 'manual start required',armed=self.armed)
            try:
                pose=self.fresh('pose',now);cmd=self.fresh('command',now);speed=self.fresh('speed',now)
                if self.armed:
                    if self.fresh('valid',now) is not True or pose is None or speed is None:
                        raise ValueError('localization/control/speed missing or stale')
                    if not math.isfinite(speed) or speed<0:raise ValueError('invalid speed')
                    self.consume_exit_gps_rejoin(pose,now)
                    progress,error=self.track(pose,now)
                    tangent=self.route.delta[self.route.index]/self.route.lengths[self.route.index]
                    heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
                    alignment=float(tangent@heading)
                    lateral=abs(float(tangent[0]*heading[1]-tangent[1]*heading[0]))
                    front_offset=max(0.,self.front_axle*alignment+self.half_width*lateral)
                    if in_ranges(self.route.index,self.passthrough):
                        # Another follower owns this segment: forward it as is.
                        passthrough=self.fresh('passthrough_command',now)
                        if passthrough is None:raise ValueError('localization/control/speed missing or stale')
                        self.in_passthrough=True
                        output=copy.deepcopy(passthrough);cap=float('inf')
                        detail=dict(state='passthrough',reason='checkpoint segment owned by pass-through follower',
                                    armed=True,route_s=progress,cross_track_error=error,speed_kmh=speed*3.6)
                    else:
                        if self.in_passthrough:
                            # Progress jumped across the segment; start the
                            # signal state fresh instead of faulting.
                            self.decision.reset();self.in_passthrough=False
                        if cmd is None:raise ValueError('localization/control/speed missing or stale')
                        signals=self.fresh('signals',now,1.)
                        if signals and now>signals['deadline']:signals=None
                        ros_now=(rospy.Time.now().to_sec()
                                 if self.decision.stoplines is not None else None)
                        cap,detail=self.decision.decide(progress,signals,now,front_offset,
                            # The GPS route turns right on green; obstacles on it
                            # belong to the route avoidance planner.
                            yield_clear=True,pose=pose,
                            speed_mps=speed,ros_now=ros_now)
                        detail.update(armed=True,route_s=progress,cross_track_error=error,speed_kmh=speed*3.6)
                        if cap>0:
                            if cmd.longlCmdType!=1 or not all(math.isfinite(v) for v in (cmd.accel,cmd.brake,cmd.steering)) or abs(cmd.steering)>math.radians(40):
                                raise ValueError('invalid raw PP/PID command')
                            output=copy.deepcopy(cmd);output.velocity=output.acceleration=0.
                            output.accel=max(0.,min(1.,cmd.accel));output.brake=max(0.,min(1.,cmd.brake))
                            if speed>=cap:
                                output.accel=0.;output.brake=max(output.brake,min(1.,max(.15,(speed-cap)/2.)))
                    detail['next_controlled_stop_distance_m'] = (
                        self.decision.distance_to_next_controlled_stop(progress,front_offset))
                    detail['route_index']=self.route.index
                    detail['route_max_error_m']=self.corridor()
                self.pose_fault_since=None
            except Exception as exc:
                # A brief GPS-to-ICP handoff may withhold the driving pose.
                # Brake throughout the gap, then resume only after every
                # ordinary route/signal/command check succeeds again.
                transient = (self.armed and self.pose_fault_grace_s>0 and
                             str(exc)=='localization/control/speed missing or stale')
                if transient:
                    if self.pose_fault_since is None:self.pose_fault_since=now
                    transient = now-self.pose_fault_since<=self.pose_fault_grace_s
                if not transient:
                    self.armed=False;self.decision.reset();self.route.index=None
                    self.pose_fault_since=None;self.ready_since=None
                cap=0.;detail=dict(state='hold',armed=self.armed,reason=str(exc))
                output=CtrlCmd();output.longlCmdType=1;output.brake=1.
            self.pub.publish(output);self.cap.publish(Float64(data=cap*3.6))
            self.camera.publish(String(data=detail.get('signal_camera','front')))
            for field in ('d_m','v_mps','signal','can_go','can_stop','decision'):
                detail.setdefault(field,None)
            self.status.publish(String(data=json.dumps(detail)))
            rospy.loginfo('Signal gate: %s',json.dumps(detail))


if __name__=='__main__':
    rospy.init_node('leo_signal_guard');SignalGuard();rospy.spin()
