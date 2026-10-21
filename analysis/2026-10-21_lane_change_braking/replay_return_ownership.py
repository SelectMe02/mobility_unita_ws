import copy,importlib.util,json,math,sys,time
from pathlib import Path
import rosbag
import yaml
from std_msgs.msg import String
ROOT=Path('/home/unita/unita_ws')
sys.path[:0]=[str(ROOT/'src/unita_waypoint/scripts'),str(ROOT/'src/unita_waypoint/test')]
import test_loop_tracking_local as tests
from local_avoidance_core_local import LocalAvoidancePlanner,Obstacle
from lane_corridor import LaneCorridor,project_lane_geometry
from map_tunner_core import primary_route,saved_lane_indices
from leo_guard.route_file import load_route_points, load_survey, load_lane_changes
from leo_avoidance.route_avoidance import RouteAvoidance
params_path=ROOT/'src/unita_waypoint/log/run_20261021_211600/rosparams.yaml'
params=yaml.safe_load(params_path.read_text())['leo_waypoint_follower']
minimum_radius=float(params.get('lane_change_min_turn_radius_m',float(params['wheelbase'])/math.tan(float(params['max_steering_rad']))))
planner_kwargs=dict(half_width=float(params.get('vehicle_half_width_m',.946)),margin=float(params.get('avoidance_margin_m',.35)),front_extent=float(params.get('front_extent_m',3.845)),rear_extent=float(params.get('rear_extent_m',.790)),preferred_lane=int(params.get('preferred_raceline',1)),min_rear_gap=float(params.get('lane_change_min_rear_gap_m',8.)),rear_headway_s=float(params.get('lane_change_rear_headway_s',2.)),reaction_s=float(params.get('avoidance_reaction_s',.4)),decel=float(params.get('avoidance_decel_mps2',3.)),pass_speed_kmh=float(params.get('pass_speed_kmh',20.)),min_turn_radius_m=minimum_radius,arc_lane_change_max_kmh=float(params.get('arc_lane_change_max_kmh',15.)),lane_change_lateral_accel_mps2=float(params.get('lane_change_lateral_accel_mps2',6.)),r1_tracking_tolerance_m=float(params.get('r1_tracking_tolerance_m',.5)))
project=json.loads((ROOT/'src/unita_waypoint/log/run_20261021_211600/config/map_tunner_project.json').read_text())
route,sectors=primary_route(project,raceline1_only=True);xy=[(p['x'],p['y']) for p in route]
lanes={n:[(p['x'],p['y']) for s in project['sectors'] for p in s['racelines'].get(str(n),[])] for n in (2,3)}
corridor=LaneCorridor(project_lane_geometry(project,float(params['effective_raceline_width']),True));states={};samples=[]
with rosbag.Bag(str(ROOT/'src/unita_waypoint/log/semi_leo_2026-10-21-21-16-00.bag')) as bag:
 start=bag.get_start_time()
 for topic,m,t in bag.read_messages(topics=['/leo/waypoint_debug/tracking_state','/leo/waypoint_debug/obstacles','/leo/route_avoidance','/competition/ego_speed_kmh']):
  states[topic]=m.data if topic.endswith('ego_speed_kmh') else json.loads(m.data)
  if topic.endswith('tracking_state') and states[topic].get('stop_reason')=='planned_speed_zero' and 89<t.to_sec()-start<91:
   samples.append((t.to_sec()-start,copy.deepcopy(states)))
spec=importlib.util.spec_from_file_location('return_before',ROOT/'rollback/2026-10-21_static_avoidance_smooth_before/src/unita_waypoint/scripts/gps_waypoint_follower_local.py');before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
results=[]
for elapsed,state in samples:
 tr=state['/leo/waypoint_debug/tracking_state'];obs=state['/leo/waypoint_debug/obstacles']['obstacles'];row={'elapsed_s':elapsed,'recorded_transition':tr['transition'],'recorded_route_state':state['/leo/route_avoidance']['state'],'recorded_route_cap_kmh':state['/leo/route_avoidance']['speed_cap_kmh']}

 # The old bag predates the independent traffic-cap field. Recompute that
 # field with the new producer's core, using its recorded projected objects
 # (rounded by its JSON publisher), real route/survey, and permitted signal.
 points=load_route_points(str(ROOT/'src/unita_waypoint/log/run_20261021_211600/config/map_tunner_project.json'))[:, :2]
 crosswalks=[(v['s']+1.,v['s']+10.) for v in load_survey(str(ROOT/'src/leo_guard/config/intersections_local.json'),points) if v.get('controlled') is True]
 rp=RouteAvoidance(points,lane_changes=load_lane_changes(str(ROOT/'src/leo_avoidance/config/route_lane_changes.json'),points),crosswalks=crosswalks)
 recorded_route=copy.deepcopy(state['/leo/route_avoidance'])
 rp._to_route=lambda obstacles,s_ego:copy.deepcopy(recorded_route['obstacles'])
 rd=rp.update((tr['x'],tr['y']),state['/competition/ego_speed_kmh']/3.6,[],stop_line_permitted=True)
 traffic_cap=None if math.isinf(rd.traffic_speed_cap_mps) else rd.traffic_speed_cap_mps*3.6
 assert traffic_cap is None
 row['new_producer_traffic_cap_evidence']={'traffic_speed_cap_kmh':traffic_cap,'all_recorded_route_objects_static':all(not o['moving'] for o in recorded_route['obstacles']),'route_ego_s':rp.last_s,'nearby_crosswalks':[(a,b) for a,b in crosswalks if rp.last_s-15<a<rp.last_s+60],'recorded_route_objects':recorded_route['obstacles'],'signal_permitted':True,'method':'Independent cap recomputed using current route producer core with recorded projected obstacle rows, actual route and survey; bag JSON positions are rounded.'}
 for name,cls in [('before',before.GPSWaypointFollower),('after_with_legacy_producer',tests.module.GPSWaypointFollower),('after',tests.module.GPSWaypointFollower)]:
  obj=tests.LoopTrackingTests().follower(xy);obj.__class__=cls
  obj.current_x,obj.current_y,obj.current_yaw=tr['x'],tr['y'],tr['yaw'];obj.current_speed_kmh=state['/competition/ego_speed_kmh'];obj.current_waypoint_index=tr['waypoint_index'];obj.first_nearest_search=False;obj.enable_local_avoidance=True
  for key in ('wheelbase','max_steering_rad','steering_sign','lookahead_distance','min_lookahead_distance','max_lookahead_distance','lookahead_speed_gain'):
   setattr(obj,key,float(params[key]))
  obj.local_planner=LocalAvoidancePlanner(xy,sectors,lanes,lane_indices=saved_lane_indices(project,route,sectors),**planner_kwargs)
  obj.local_planner.transition=copy.deepcopy(tr['transition']);obj.local_planner.active_lane=None
  obj.obstacles=[Obstacle(*(o[k] for k in ('x','y','radius','vx','vy')),kind=o['kind']) for o in obs]
  obj.obstacles=[o for o,inside in zip(obj.obstacles,corridor.contains([(o.x,o.y) for o in obj.obstacles])) if inside]
  obj.obstacle_received=time.monotonic();obj.obstacle_stamp=100.
  obj.route_avoidance_topic='/leo/route_avoidance'
  plan=copy.deepcopy(recorded_route)
  if name=='after':plan['traffic_speed_cap_kmh']=traffic_cap
  obj.route_avoidance_callback(String(data=json.dumps(plan)))
  tests.LoopTrackingTests().run_loop(obj)
  row[name]={'stop_reason':obj.stop_reason,'local_path_source':obj.debug_trace['local_path_source'],'planner_state':obj.debug_trace.get('planner_state'),'target_speed_kmh':obj.commands[-1][1] if obj.commands else 0.,'full_stop':bool(obj.ctrl_pub.publish.called),'transition':obj.local_planner.transition}
 results.append(row)
out={'source_bag':'semi_leo_2026-10-21-21-16-00.bag','method':'Run old and new follower control loops offline at recorded planned_speed_zero return sample, with recorded pose/speed/obstacles/transition/route plan and actual saved-lane geometry. Stub publishers only; clocks normalized to the fresh sample. New-producer independent traffic cap is recomputed separately using recorded route obstacles and actual map/survey; a third run checks conservative compatibility with the original legacy producer.','settings_source':str(params_path),'recorded_wheelbase_m':params['wheelbase'],'recorded_max_steering_rad':params['max_steering_rad'],'resolved_min_turn_radius_m':minimum_radius,'local_planner_parameters':planner_kwargs,'rows':results}
p=ROOT/'analysis/2026-10-21_lane_change_braking/return_ownership_fix_replay.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
