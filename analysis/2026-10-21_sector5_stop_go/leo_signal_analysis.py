import rosbag,json,collections,datetime
p='src/unita_waypoint/log/semi_leo_2026-10-19-00-25-29.bag'
T=['/leo/waypoint_debug/tracking_state','/leo/signal_status','/leo/route_avoidance','/competition/ego_speed_kmh','/leo/raw_ctrl_cmd','/ctrl_cmd','/leo/waypoint_debug/target_speed_kmh']
latest={};prev=None;rows=[]; out={}
with rosbag.Bag(p) as b:
 start=b.get_start_time()
 for topic,m,t in b.read_messages(topics=T):
  ts=t.to_sec();d=json.loads(m.data) if topic in T[:3] else (m.data if hasattr(m,'data') else {'brake':m.brake,'accel':m.accel,'steering':m.steering})
  latest[topic]=d
  tr=latest.get(T[0],{});sec=tr.get('sector')
  if topic!=T[1]: continue
  sig=d;raw=latest.get(T[4],{});cmd=d.get('command') or {}; inp=d.get('input_command') or {};route=latest.get(T[2],{});v=latest.get(T[3],-1)
  row={'t':round(ts-start,3),'epoch':ts,'local':datetime.datetime.fromtimestamp(ts,datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'sec':sec,'wp':tr.get('waypoint_index'),'xy':[round(tr.get('x',0),2),round(tr.get('y',0),2)],'state':tr.get('driving_state'),'lane':tr.get('raceline'),'current':tr.get('current_raceline'),'why':tr.get('lane_change_reason'),'obst':tr.get('obstacle_distance_m'),'kind':tr.get('obstacle_kind'),'v':round(v,3),'raw_brake':raw.get('brake'),'raw_accel':raw.get('accel'),'in_brake':inp.get('brake'),'out_brake':cmd.get('brake'),'in_accel':inp.get('accel'),'out_accel':cmd.get('accel'),'sig_state':sig.get('state'),'sig_dec':sig.get('decision'),'sig':sig.get('signal'),'sig_reason':sig.get('reason'),'sig_cap':sig.get('speed_cap_kmh'),'sig_dist':sig.get('d_m'),'sig_id':sig.get('traffic_light_id'),'sig_next':sig.get('next_controlled_stop_distance_m'),'route_state':route.get('state'),'route_reason':route.get('reason'),'route_cap':route.get('speed_cap_kmh')}
  rows.append(row)
with open('/tmp/leo_signal_rows.jsonl','w') as f:
 for r in rows:f.write(json.dumps(r)+'\n')
for sec in [4,5,6]:
 subset=[r for r in rows if r['sec']==sec]
 print('SECTION',sec,'n',len(subset),'minv',min(r['v'] for r in subset),'signal zero',sum(r['sig_cap']==0 for r in subset),'signal injected full brake',sum(r['out_brake']==1 and r['in_brake']!=1 for r in subset))
 print('STATES',dict(collections.Counter((r['sig_state'],r['sig_dec'],r['sig']) for r in subset)))
 prev=None
 for r in subset:
  key=(r['state'],r['why'],r['in_brake']==1,r['out_brake']==1,r['sig_state'],r['sig_dec'],r['v']<.5)
  if key!=prev and (sec==5 or r['in_brake']==1 or r['out_brake']==1 or r['v']<.5):
   print(json.dumps(r))
  prev=key
