"""Offline queue/synchronization scheduling replay; no ROS nodes or publishers."""
import importlib.util,json,math,sys,threading,statistics
from pathlib import Path
from collections import deque
from types import SimpleNamespace
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2

spec=importlib.util.spec_from_file_location('route_queue_replay','src/leo_avoidance/scripts/route_avoidance_node.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
data=json.load(open(Path(__file__).with_name('route_timing_inputs.json')))
start=data['clouds'][0]['at']-data['clouds'][0]['t'];end=max(x['at'] for x in data['/leo/driving_pose'])
clock=[start];module.time.monotonic=lambda:clock[0]
module.rospy.Time.now=lambda:rospy.Time.from_sec(clock[0])

def node():
 o=module.RouteAvoidanceNode.__new__(module.RouteAvoidanceNode)
 o.lock=threading.Lock();o.poses=deque(maxlen=100);o.pose=None;o.max_pose_gap_s=.15;o.input_timeout_s=.35
 o.pending=deque(maxlen=8);o.last_selected_stamp=None;o.input_epoch=o.processed_epoch=0;o.last_worker_ros_time=None;o.closed=False
 o.tracker=SimpleNamespace(reset=lambda:None);o.planner=SimpleNamespace(reset=lambda:None)
 return o

def input_events():
 result=[]
 for x in data['/leo/driving_pose']:
  m=PoseStamped();m.header.frame_id='map';m.header.stamp=rospy.Time.from_sec(x['stamp']);m.pose.position.x=x['x'];m.pose.position.y=x['y'];m.pose.orientation.z=math.sin(x['yaw']/2);m.pose.orientation.w=math.cos(x['yaw']/2)
  result.append((x['at'],0,'pose',m))
 for x in data['clouds']:
  m=PointCloud2();m.header.frame_id='lidar';m.header.stamp=rospy.Time.from_sec(x['stamp']);result.append((x['at'],1,'cloud',m))
 return result
inputs=input_events()

def replay(phase,queue,processing_s=.01):
 o=node();pending=None;outputs=[];occupied_until=0.
 events=inputs+[(t,2,'tick',None) for t in np.arange(start+phase,end,.02)]
 for t,_,kind,message in sorted(events,key=lambda e:(e[0],e[1])):
  clock[0]=t
  if kind=='pose':o.on_pose(message)
  elif kind=='cloud':
   if queue:o.on_cloud(message)
   else:pending=(message,t)
  elif t>=occupied_until:
   if queue:
    selected=o.take_ready_scan(t,t)
    if selected:message,received,pose,_=selected
   else:
    selected=None
    if pending:
     message,received=pending;pending=None;stamp=message.header.stamp.to_sec()
     if -.05<=t-stamp<=.35 and 0<=t-received<=.35:
      pose=o.pose_at(stamp)
      if pose is not None:selected=(message,received,pose,0)
      else:pending=(message,received)
   if selected:
    stamp=message.header.stamp.to_sec();finish=t+processing_s;occupied_until=finish
    if finish-stamp<=.35 and finish-received<=.35:
     outputs.append(dict(t=finish-start,stamp=stamp,age=finish-stamp))
 gaps=[dict(start=a['t'],end=b['t'],gap=b['t']-a['t']) for a,b in zip(outputs,outputs[1:])]
 relevant=[g for g in gaps if g['start']<94 and g['end']>85.45]
 return dict(phase_ms=phase*1000,processing_ms=processing_s*1000,outputs=len(outputs),max_age=max(o['age'] for o in outputs),max_gap=max(g['gap'] for g in gaps),gaps_over_timeout=[g for g in gaps if g['gap']>.5],avoidance_max_gap=max(g['gap'] for g in relevant),avoidance_gaps_over_timeout=[g for g in relevant if g['gap']>.5],output_times=outputs)

results={}
for kind,queued in [('old_single_pending',False),('new_ready_queue',True)]:
 rows=[replay(ms/1000.,queued) for ms in range(20)]
 results[kind]=rows
 print(kind,'outputrange',min(r['outputs'] for r in rows),max(r['outputs'] for r in rows),'gaprange',min(r['max_gap'] for r in rows),max(r['max_gap'] for r in rows),'avoidancemaxgap',max(r['avoidance_max_gap'] for r in rows),'avoidance_stale_gapcount_range',min(len(r['avoidance_gaps_over_timeout']) for r in rows),max(len(r['avoidance_gaps_over_timeout']) for r in rows))
 print('phase0', {k:v for k,v in rows[0].items() if k!='output_times'})
for processing in [.0,.02,.04]:
 row=replay(0.,True,processing);results['new_compute_'+str(processing)]=row
 print('compute_s',processing,'outputs',row['outputs'],'avoidance_max_gap',row['avoidance_max_gap'],'avoidance_gapcount',len(row['avoidance_gaps_over_timeout']))
Path(__file__).with_name('route_queue_replay.json').write_text(json.dumps(results,indent=2))
