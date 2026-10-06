#!/usr/bin/env python3
"""Camera-independent observed LiDAR occupancy; never publishes vehicle control."""
import collections
import json
import math
import threading
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String
from leo_guard.traffic import RouteProgress
from leo_guard.yield_check import corridor_observation


class YieldObserver:
    def __init__(self):
        self.lock=threading.Lock();self.pending=None;self.poses=collections.deque(maxlen=80)
        self.route=RouteProgress(np.loadtxt(rospy.get_param('~waypoint_file')),
                                 backward_window=30)
        # Measured in the loaded LEO sensor file; zero roll/pitch/yaw.
        self.translation=np.asarray(rospy.get_param('~lidar_to_rear_xyz'),float)
        if self.translation.shape!=(3,) or not np.isfinite(self.translation).all():raise ValueError('invalid measured LiDAR transform')
        self.pub=rospy.Publisher('/leo/yield_observation',String,queue_size=1)
        self.subs=[rospy.Subscriber('/localization/pose',PoseStamped,self.pose,queue_size=1),
                   rospy.Subscriber('/lidar3D',PointCloud2,self.cloud,queue_size=1,buff_size=2**24)]
        threading.Thread(target=self.worker,daemon=True).start()

    def pose(self,m):
        q=m.pose.orientation
        value=(m.header.stamp.to_sec(),m.pose.position.x,m.pose.position.y,
               math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        if m.header.frame_id=='map' and all(math.isfinite(v) for v in value):
            with self.lock:self.poses.append(value)

    def cloud(self,m):
        with self.lock:self.pending=m

    def worker(self):
        while not rospy.is_shutdown():
            with self.lock:m=self.pending;self.pending=None;poses=list(self.poses)
            if m is None:time.sleep(.01);continue
            stamp=m.header.stamp.to_sec();result=dict(valid=False,clear=False,reason='pose unavailable')
            try:
                age=rospy.Time.now().to_sec()-stamp
                if m.header.frame_id!='lidar' or not 0<=age<=.35 or not poses:raise ValueError('cloud/pose missing or stale')
                pose=min(poses,key=lambda p:abs(p[0]-stamp))
                if abs(pose[0]-stamp)>.12:raise ValueError('unmatched capture-time pose')
                _,x,y,yaw=pose;self.route.locate(x,y,yaw)
                i=self.route.index;r=self.route.points[i:min(i+75,len(self.route.points))]
                c,s=math.cos(yaw),math.sin(yaw)
                r=(r-[x,y])@np.array([[c,-s],[s,c]])
                p=np.asarray(list(point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)),float)
                result=corridor_observation(p+self.translation,r)
                if rospy.Time.now().to_sec()-stamp>.35:raise ValueError('cloud processing stale')
            except Exception as exc:result=dict(valid=False,clear=False,reason=str(exc))
            result.update(stamp=stamp,frame=stamp)
            self.pub.publish(String(data=json.dumps(result)))


if __name__=='__main__':
    rospy.init_node('leo_yield_observer');YieldObserver();rospy.spin()
