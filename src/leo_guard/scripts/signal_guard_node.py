#!/usr/bin/env python3
"""Camera signal gate for PP/PID; lane detection is not a drive prerequisite."""
import copy
import json
import math
import threading
import time
from pathlib import Path
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from morai_msgs.msg import CtrlCmd
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import SetBool, SetBoolResponse
from leo_guard.traffic import RouteProgress, TrafficDecision
from leo_guard.core import SignalLatch


class SignalGuard:
    def __init__(self):
        self.lock=threading.RLock();self.inputs={};self.armed=False
        self.yield_latch=SignalLatch(1.,.35)
        self.stop_since=None;self.stop_id=None;self.stop_completed=False
        self.pose_fault_since=None
        self.pose_fault_grace_s=float(rospy.get_param('~pose_fault_grace_s',0.0))
        if not math.isfinite(self.pose_fault_grace_s) or self.pose_fault_grace_s<0:
            raise ValueError('invalid pose fault grace')
        points=np.loadtxt(rospy.get_param('~waypoint_file'))
        survey=json.loads(Path(rospy.get_param('~intersection_file')).read_text())
        stoplines=json.loads(Path(rospy.get_param('~stoplines_file')).read_text())
        self.route=RouteProgress(points,
            max_error=float(rospy.get_param('~route_max_error_m',2.0)),
            backward_window=30)
        # GPS sensor transform (0,0,1.2) was inspected in the simulator.
        # MORAI sensor origin is the rear axle; competition wheelbase is 3 m.
        self.front_axle=float(rospy.get_param('~gps_to_front_axle_m',0.))
        self.half_width=float(rospy.get_param('~vehicle_half_width_m',.946))
        if not all(math.isfinite(v) and 0<=v<=5. for v in (self.front_axle,self.half_width)):
            raise ValueError('invalid measured vehicle/GPS geometry')
        # The file is derived from this exact route, not an unrelated map.
        import hashlib
        digest=hashlib.sha256(Path(rospy.get_param('~waypoint_file')).read_bytes()).hexdigest()
        if survey.get('route_sha256')!=digest:raise ValueError('intersection survey route hash mismatch')
        self.decision=TrafficDecision(survey['intersections'],self.route.total,
            front_margin=float(rospy.get_param('~stop_margin_m',5.)),
            maximum_kmh=float(rospy.get_param('~maximum_speed_kmh',58.)),
            stoplines=stoplines)
        self.pub=rospy.Publisher('/ctrl_cmd',CtrlCmd,queue_size=1)
        self.cap=rospy.Publisher('/leo/speed_limit_kmh',Float64,queue_size=1)
        self.status=rospy.Publisher('/leo/signal_status',String,queue_size=1)
        self.camera=rospy.Publisher('/leo/signal_camera',String,queue_size=1,latch=True)
        self.subs=[rospy.Subscriber('/leo/raw_ctrl_cmd',CtrlCmd,lambda m:self.put('command',m),queue_size=1),
            rospy.Subscriber('/localization/pose',PoseStamped,self.pose,queue_size=1),
            rospy.Subscriber('/localization/valid',Bool,lambda m:self.put('valid',m.data),queue_size=1),
            rospy.Subscriber('/competition/ego_speed_kmh',Float64,lambda m:self.put('speed',m.data/3.6),queue_size=1),
            rospy.Subscriber('/perception/traffic_lights',String,self.signals,queue_size=1),
            rospy.Subscriber('/leo/yield_observation',String,self.yield_observation,queue_size=1)]
        self.service=rospy.Service('~set_armed',SetBool,self.set_armed)
        self.timer=rospy.Timer(rospy.Duration(.05),self.tick)
        rospy.logwarn('Signal practice starts with brake held. Survey/head review is required before intersection release.')

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

    def fresh(self,key,now,timeout=.35):
        item=self.inputs.get(key)
        return item[0] if item and 0<=now-item[1]<=timeout else None

    def yield_observation(self,m):
        value=None
        try:
            v=json.loads(m.data);age=rospy.Time.now().to_sec()-float(v['stamp'])
            if math.isfinite(age) and 0<=age<=.35:
                v['deadline']=time.monotonic()+.35-age;value=v
        except (ValueError,KeyError,TypeError):pass
        self.put('yield',value)

    def set_armed(self,req):
        with self.lock:
            now=time.monotonic()
            if req.data:
                pose=self.fresh('pose',now)
                missing=[]
                if self.fresh('valid',now) is not True:missing.append('driving_valid')
                if pose is None:missing.append('driving_pose')
                if self.fresh('speed',now) is None:missing.append('ego_speed')
                if missing:
                    return SetBoolResponse(False,'fresh input required: '+', '.join(missing))
                try:self.route.locate(*pose)
                except ValueError as exc:return SetBoolResponse(False,str(exc))
            self.armed=bool(req.data);self.decision.reset()
            self.stop_since=None;self.stop_id=None;self.stop_completed=False
            self.pose_fault_since=None
            self.yield_latch.update(False,None,None,now)
        return SetBoolResponse(True,'armed' if self.armed else 'brake held')

    def tick(self,event):
        output=CtrlCmd();output.longlCmdType=1;output.brake=1.;cap=0.
        detail=dict(state='hold',reason='manual start required',armed=self.armed)
        with self.lock:
            now=time.monotonic()
            try:
                pose=self.fresh('pose',now);cmd=self.fresh('command',now);speed=self.fresh('speed',now)
                if self.armed:
                    if self.fresh('valid',now) is not True or pose is None or cmd is None or speed is None:
                        raise ValueError('localization/control/speed missing or stale')
                    if not math.isfinite(speed) or speed<0:raise ValueError('invalid speed')
                    progress,error=self.route.locate(*pose)
                    signals=self.fresh('signals',now,1.)
                    if signals and now>signals['deadline']:signals=None
                    tangent=self.route.delta[self.route.index]/self.route.lengths[self.route.index]
                    heading=np.array([math.cos(pose[2]),math.sin(pose[2])])
                    alignment=float(tangent@heading)
                    lateral=abs(float(tangent[0]*heading[1]-tangent[1]*heading[0]))
                    front_offset=max(0.,self.front_axle*alignment+self.half_width*lateral)
                    observation=self.fresh('yield',now)
                    observed=bool(observation and observation.get('valid') is True and observation.get('clear') is True and now<=observation['deadline'])
                    stable=self.yield_latch.update(observed,'corridor',observation.get('frame') if observation else None,now)
                    active=self.decision.active
                    if active and active[1]['id']!=self.stop_id:
                        self.stop_id=active[1]['id'];self.stop_since=None;self.stop_completed=False
                    if active and active[1]['maneuver']=='right_unprotected' and active[0]<8 and speed<.2:
                        if self.stop_since is None:self.stop_since=now
                        if now-self.stop_since>=.5:self.stop_completed=True
                    elif not self.stop_completed:self.stop_since=None
                    ros_now=(rospy.Time.now().to_sec()
                             if self.decision.stoplines is not None else None)
                    cap,detail=self.decision.decide(progress,signals,now,front_offset,
                        yield_clear=stable and self.stop_completed,pose=pose,
                        speed_mps=speed,ros_now=ros_now)
                    if detail.get('maneuver')=='right_unprotected' or (self.decision.committed and self.decision.committed[0]['maneuver']=='right_unprotected'):
                        detail.update(yield_observation=observation,stop_completed=self.stop_completed)
                        if not observed and (detail.get('stop_distance',0)<=8 or detail['state']=='clearing'):
                            cap=0.;detail['reason']='right-turn observed corridor missing or occupied'
                    detail.update(armed=True,route_s=progress,cross_track_error=error,speed_kmh=speed*3.6)
                    detail['next_controlled_stop_distance_m'] = (
                        self.decision.distance_to_next_controlled_stop(progress,front_offset))
                    if cap>0:
                        if cmd.longlCmdType!=1 or not all(math.isfinite(v) for v in (cmd.accel,cmd.brake,cmd.steering)) or abs(cmd.steering)>math.radians(40):
                            raise ValueError('invalid raw PP/PID command')
                        output=copy.deepcopy(cmd);output.velocity=output.acceleration=0.
                        output.accel=max(0.,min(1.,cmd.accel));output.brake=max(0.,min(1.,cmd.brake))
                        if speed>=cap:
                            output.accel=0.;output.brake=max(output.brake,min(1.,max(.15,(speed-cap)/2.)))
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
                    self.pose_fault_since=None
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
