#!/usr/bin/env python3
"""Classical white/yellow paint candidates; no learned lane model required."""
import json
import threading
import time
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String
from leo_guard.core import project_ground_pixels,extract_scene

class CameraLane:
    def __init__(self):
        self.bridge=CvBridge();self.lock=threading.Lock();self.latest=None
        self.verified=rospy.get_param('~calibration_verified',False)
        self.k=rospy.get_param('~intrinsics',[]);self.t=rospy.get_param('~optical_to_rear',[])
        self.pub=rospy.Publisher('/leo/camera_lane_candidates',String,queue_size=1)
        self.sub=rospy.Subscriber('/camera/image/front',Image,self.receive,queue_size=1,buff_size=2**24)
    def receive(self,msg):
        with self.lock:self.latest=msg
    def run(self):
        while not rospy.is_shutdown():
            with self.lock:msg=self.latest;self.latest=None
            if msg:
                scene={'valid':False,'reason':'camera calibration not verified'}
                try:
                    image=self.bridge.imgmsg_to_cv2(msg,'bgr8');hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                    white=cv2.inRange(hsv,np.array([0,0,180]),np.array([179,65,255]))
                    yellow=cv2.inRange(hsv,np.array([15,70,90]),np.array([40,255,255]))
                    mask=cv2.bitwise_or(white,yellow);mask[:int(image.shape[0]*.4)]=0
                    y,x=np.nonzero(mask);step=max(1,len(x)//12000);pixels=np.c_[x[::step],y[::step]]
                    scene['pixel_candidates']=len(pixels)
                    age=rospy.Time.now().to_sec()-msg.header.stamp.to_sec()
                    if self.verified and -.1<=age<=.35:
                        p=project_ground_pixels(pixels,self.k,self.t)
                        scene=extract_scene(np.c_[p,np.full(len(p),110.)],0.)
                        if scene.get('valid'):
                            for side in ('left','right'):scene[side]['uncertainty']=max(.25,scene[side]['uncertainty'])
                    elif self.verified:scene['reason']='stale camera source'
                except Exception as exc:scene={'valid':False,'reason':'camera candidate error: '+str(exc)}
                scene['stamp']=msg.header.stamp.to_sec();scene['source']='camera_geometric'
                self.pub.publish(String(data=json.dumps(scene)))
            time.sleep(.1)

if __name__=='__main__':
    rospy.init_node('leo_camera_lane');CameraLane().run()
