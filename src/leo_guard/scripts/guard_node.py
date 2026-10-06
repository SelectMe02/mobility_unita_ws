#!/usr/bin/env python3
"""Experimental single-writer command guard; missing evidence holds the brake.

Traffic context requires route-associated, reviewed stop-line perception. This
node intentionally cannot infer the correct signal from largest image box.
"""
import copy
import json
import math
import threading
import time
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64, String, Bool
from morai_msgs.msg import CtrlCmd
from leo_guard.core import extract_scene, constrain_command, signal_permission, SignalLatch


class Guard:
    def __init__(self):
        self.lock=threading.Lock();self.inputs={};self.scene={};self.pending=None
        self.approved=all(rospy.get_param('~'+k,False) for k in (
            'calibration_verified','lane_perception_verified','traffic_association_verified'))
        self.camera_verified=rospy.get_param('~camera_fallback_verified',False)
        self.geometry={k:float(rospy.get_param('~'+k)) for k in (
            'wheelbase','front_extent','rear_overhang','half_width')}
        self.timeout=float(rospy.get_param('~input_timeout',.35))
        self.latency=float(rospy.get_param('~reaction_latency',.4))
        self.decel=float(rospy.get_param('~assumed_deceleration',3.))
        self.margin=float(rospy.get_param('~clearance_margin',.25))
        self.signal_timeout=float(rospy.get_param('~signal_timeout',1.))
        self.context_timeout=float(rospy.get_param('~context_timeout',.35))
        self.roi=rospy.get_param('~signal_roi')
        self.xyz=np.array(rospy.get_param('~lidar_to_rear_xyz'),float)
        roll,pitch,yaw=np.radians(rospy.get_param('~lidar_to_rear_rpy_deg'))
        cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
        self.rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                                [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.rotation).all():raise ValueError('invalid calibration')
        if any(not math.isfinite(v) or v<=0 for v in list(self.geometry.values())+[
                self.timeout,self.latency,self.decel,self.margin,self.signal_timeout,self.context_timeout]):
            raise ValueError('invalid guard configuration')
        self.latch=SignalLatch(float(rospy.get_param('~signal_stable_seconds',.6)),self.signal_timeout)
        self.pub=rospy.Publisher('/ctrl_cmd',CtrlCmd,queue_size=1)
        self.status=rospy.Publisher('/leo/guard_status',String,queue_size=1)
        self.cap=rospy.Publisher('/leo/speed_limit_kmh',Float64,queue_size=1)
        self.scene_pub=rospy.Publisher('/leo/lane_candidates',String,queue_size=1)
        self.subs=[rospy.Subscriber('/leo/raw_ctrl_cmd',CtrlCmd,lambda m:self.put('command',m),queue_size=1),
                   rospy.Subscriber('/competition/ego_speed_kmh',Float64,lambda m:self.put('speed',m.data/3.6),queue_size=1),
                   rospy.Subscriber('/localization/valid',Bool,lambda m:self.put('localization',m.data),queue_size=1),
                   rospy.Subscriber('/perception/traffic_lights',String,lambda m:self.json_input('signals',m),queue_size=1),
                   rospy.Subscriber('/leo/traffic_context',String,lambda m:self.json_input('traffic',m),queue_size=1),
                   rospy.Subscriber('/leo/camera_lane_candidates',String,lambda m:self.json_input('camera_lane',m),queue_size=1),
                   rospy.Subscriber('/lidar3D',PointCloud2,self.receive_cloud,queue_size=1,buff_size=2**24)]
        self.worker=threading.Thread(target=self.cloud_worker,daemon=True);self.worker.start()
        self.timer=rospy.Timer(rospy.Duration(.05),self.tick)

    def put(self,key,value):
        with self.lock:self.inputs[key]=(value,time.monotonic())

    def json_input(self,key,message):
        try:
            value=json.loads(message.data)
            stamp=float(value['stamp']);age=rospy.Time.now().to_sec()-stamp
            if not math.isfinite(age) or age<-.1 or age>(self.signal_timeout if key=='signals' else self.context_timeout):
                raise ValueError('stale/future source stamp')
            value['_source_deadline']=time.monotonic()+max(0.,(self.signal_timeout if key=='signals' else self.context_timeout)-age)
            self.put(key,value)
        except (ValueError,KeyError,TypeError):self.put(key,None)

    def receive_cloud(self,message):
        with self.lock:self.pending=(message,time.monotonic())

    def cloud_worker(self):
        while not rospy.is_shutdown():
            with self.lock:item=self.pending;self.pending=None
            if item:
                try:
                    msg,received=item
                    age=rospy.Time.now().to_sec()-msg.header.stamp.to_sec()
                    if not math.isfinite(age) or age<-.1 or age>self.timeout:raise ValueError('stale cloud stamp')
                    p=np.array(list(point_cloud_2 for point_cloud_2 in point_cloud2.read_points(
                        msg,field_names=('x','y','z','intensity'),skip_nans=True)),dtype=float)
                    if len(p):p[:,:3]=p[:,:3]@self.rotation.T+self.xyz
                    scene=extract_scene(p,float(rospy.get_param('~ground_z_rear')),
                                        intensity_floor=float(rospy.get_param('~intensity_floor',65.)))
                    # Timestamp is reception, not completion: slow fitting is stale.
                    with self.lock:self.inputs['lane']=(scene,received-max(0.,age))
                    self.scene_pub.publish(String(data=json.dumps(scene)))
                except Exception as exc:self.put('lane',{'valid':False,'reason':str(exc)})
            time.sleep(.01)

    def tick(self,event):
        output=CtrlCmd();output.longlCmdType=1;output.accel=0.;output.brake=1.;output.steering=0.
        permitted=False;cap=0.;reason='inputs unavailable'
        try:
            with self.lock:items=dict(self.inputs)
            now=time.monotonic()
            def fresh(key,timeout):
                item=items.get(key)
                return item[0] if item and now-item[1]<=timeout else None
            cmd=fresh('command',self.timeout);speed=fresh('speed',self.timeout)
            scene=fresh('lane',self.timeout);loc=fresh('localization',self.timeout)
            camera=fresh('camera_lane',self.context_timeout)
            # A camera fallback must not bypass unavailable LiDAR obstacle input.
            if (getattr(self,'camera_verified',False) and scene and scene.get('ground_valid')
                    and not scene.get('valid') and camera and camera.get('valid')
                    and now<=camera.get('_source_deadline',0)):
                camera=copy.deepcopy(camera);camera['obstacle_distance']=scene.get('obstacle_distance')
                scene=camera
            traffic=copy.deepcopy(fresh('traffic',self.context_timeout));signals=fresh('signals',self.signal_timeout)
            if cmd is not None and speed is not None and math.isfinite(speed) and speed>=0 and scene and loc is True:
                if traffic and now>traffic.get('_source_deadline',0):traffic=None
                permission=False
                if traffic and traffic.get('state')=='approach':
                    if signals and signals.get('valid') and now<=signals.get('_source_deadline',0):
                        permission,label=signal_permission(signals.get('detections',[]),traffic.get('maneuver'),self.roi)
                        permission=self.latch.update(permission,(traffic.get('intersection_id'),traffic.get('maneuver'),label),signals.get('frame'),now)
                    else:self.latch.update(False,None,None,now)
                    traffic['release_stable']=permission
                else:self.latch.update(False,None,None,now)
                permitted,cap,reason=constrain_command(scene,speed,float(cmd.steering),self.geometry,
                                                      traffic,signals,self.approved,self.latency,self.decel,self.margin)
                if permitted:
                    output=copy.deepcopy(cmd);output.longlCmdType=1;output.velocity=0.;output.acceleration=0.
                    if not all(math.isfinite(v) for v in (cmd.accel,cmd.brake,cmd.steering)) or abs(cmd.steering)>math.radians(40):
                        raise ValueError('invalid raw control')
                    output.accel=max(0.,min(1.,cmd.accel));output.brake=max(0.,min(1.,cmd.brake))
                    if speed>=cap:
                        output.accel=0.;output.brake=max(output.brake,min(1.,max(.15,(speed-cap)/2.)))
        except Exception as exc:
            permitted=False;cap=0.;reason='guard error: '+str(exc)
            output=CtrlCmd();output.longlCmdType=1;output.accel=0.;output.brake=1.;output.steering=0.
        self.pub.publish(output);self.cap.publish(Float64(data=cap*3.6))
        self.status.publish(String(data=json.dumps({'permit':permitted,'speed_cap_kmh':cap*3.6,'reason':reason})))


if __name__=='__main__':
    rospy.init_node('leo_guard');guard=Guard();rospy.spin()
