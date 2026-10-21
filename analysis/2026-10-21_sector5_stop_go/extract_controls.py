import rosbag,json,collections,datetime,math
from pathlib import Path
out=Path('analysis/2026-10-21_sector5_stop_go')
bagpath='src/unita_waypoint/log/semi_leo_2026-10-19-00-25-29.bag'
T=datetime.timezone(datetime.timedelta(hours=9))
def ts(t):return datetime.datetime.fromtimestamp(t,T).isoformat(timespec='milliseconds')
rows=[];prev={};sectors=[];last_sector=None;topic_rows=collections.defaultdict(list)
with rosbag.Bag(bagpath) as b:
 for topic,msg,stamp in b.read_messages(topics=['/competition/ego_speed_kmh','/ctrl_cmd','/leo/raw_ctrl_cmd','/leo/route_avoidance','/leo/signal_status','/leo/waypoint_debug/tracking_state','/leo/waypoint_debug/target_speed_kmh','/rosout_agg']):
  t=stamp.to_sec()
  if topic=='/rosout_agg':
   topic_rows['rosout'].append(dict(t=t,time=ts(t),name=msg.name,level=msg.level,msg=msg.msg))
   continue
  if topic in ['/ctrl_cmd','/leo/raw_ctrl_cmd']:
   d={k:getattr(msg,k) for k in ['accel','brake','steering','velocity']}
  elif topic in ['/competition/ego_speed_kmh','/leo/waypoint_debug/target_speed_kmh']:d=msg.data
  else:d=json.loads(msg.data)
  if topic=='/leo/route_avoidance':d={k:v for k,v in d.items() if k!='path'}
  prev[topic]=(t,d)
  if topic=='/leo/waypoint_debug/tracking_state' and d.get('sector')!=last_sector:
   sectors.append(dict(t=t,time=ts(t),sector=d.get('sector'),x=d.get('x'),y=d.get('y'),wp=d.get('waypoint_index')))
   last_sector=d.get('sector')
  topic_rows[topic].append(dict(t=t,data=d))
  if topic!='/ctrl_cmd':continue
  r={'t':t,'time':ts(t)}
  for key,name in [('/ctrl_cmd','cmd'),('/leo/raw_ctrl_cmd','raw'),('/leo/route_avoidance','route'),('/leo/signal_status','guard'),('/leo/waypoint_debug/tracking_state','track'),('/competition/ego_speed_kmh','speed'),('/leo/waypoint_debug/target_speed_kmh','target')]:
   at,val=prev.get(key,(None,None));r[name]=val;r[name+'_age']=None if at is None else t-at
  rows.append(r)
(out/'sector_transitions.json').write_text(json.dumps(sectors,indent=2))
(out/'all_controls.json').write_text(json.dumps(rows))
for k,rs in topic_rows.items():
 name=k.strip('/').replace('/','_') or 'unknown'
 (out/(name+'.json')).write_text(json.dumps(rs))
print(json.dumps(sectors,indent=2))
s5=[r for r in rows if r['track'] and r['track'].get('sector') in [5,6]]
print('s5/6rows',len(s5))
for sector in [5,6]:
 rs=[r for r in s5 if r['track']['sector']==sector]
 if not rs:continue
 print('SECTOR',sector,'time',rs[0]['time'],rs[-1]['time'],'n',len(rs),'guard',collections.Counter((r['guard'].get('state'),r['guard'].get('reason')) for r in rs if r['guard']),'route',collections.Counter((r['route']['state'],r['route']['reason']) for r in rs if r['route']))
