import sys
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from map_tunner_core import load_project, primary_route
from local_avoidance_core import LocalAvoidancePlanner, Obstacle
from raceline_test_follower import RacelineTestFollower


class RacelineSelectionTest(unittest.TestCase):
    def test_driving_preserves_saved_geometry_even_with_leftover_width_override(self):
        from raceline_test_follower import RacelineTestFollower
        project_file = ROOT / 'config/map_tunner_project.json'
        saved = load_project(project_file)
        obj = RacelineTestFollower.__new__(RacelineTestFollower)
        for override in (None, '', 3.5, 4.0):
            with self.subTest(override=override), patch('raceline_test_follower.rospy.get_param',
                    side_effect=lambda name, default=None:
                    override if name == '~raceline_width' else default):
                self.assertEqual(obj.load_map_tunner_project(str(project_file)), saved)

    def test_screenshot_position_and_extension_with_live_width_parameter(self):
        obj = RacelineTestFollower.__new__(RacelineTestFollower)
        obj.waypoint_file = str(ROOT / 'config/map_tunner_project.json')
        obj.waypoint_format = 'local_xy'
        obj.raceline_width_override = 3.5
        obj.lane_geometry_pub = MagicMock()
        params = {'~raceline_width': 3.5, '~preferred_raceline': 3,
                  '~preferred_raceline_sector': 11, '~preferred_raceline_end_checkpoint': '11'}
        with patch('rospy.get_param', side_effect=lambda n, d=None: params.get(n, d)), \
                patch('rospy.set_param') as set_param:
            obj.load_waypoints()
            obj.waypoints = obj.raw_waypoints
            fallback, preferences = obj.local_lane_preferences()
        self.assertNotIn('~raceline_width', [call.args[0] for call in set_param.call_args_list])
        p = LocalAvoidancePlanner(obj.waypoints, obj.route_sectors, obj.project_lanes,
                                  lane_indices=obj.project_lane_indices, preferred_lane=fallback,
                                  preferred_lane_indices=preferences,
                                  finish_line=obj.lane_preference_finish_line)
        published = json.loads(obj.lane_geometry_pub.publish.call_args.args[0].data)
        r3 = next(l for l in published['lanes'] if l['sector'] == 11 and l['raceline'] == 3)
        self.assertLess(min(pt[1] for pt in r3['points']), -217.85)
        for y in (-118.38, -119., -122.9, -150., -200.):
            index = min(obj.project_lane_indices[3], key=lambda i: abs(obj.waypoints[i][1]-y))
            ego = (74.64, y) if y == -118.38 else obj.project_lane_indices[3][index]
            decision = p.plan(index, ego, 5., [], ego_yaw=math.radians(-88.5))
            self.assertEqual(p.effective_preferred_lane, 3)
            self.assertGreater(decision.speed_cap_kmh, 0.)
            self.assertEqual(decision.state, 'RACELINE')
            self.assertLess(decision.path[-1][1], y-20.)

    def test_valid_sensor_callbacks_never_temporarily_clear_ready(self):
        from gps_waypoint_follower import GPSWaypointFollower
        obj = GPSWaypointFollower.__new__(GPSWaypointFollower)
        obj.gps_ready = obj.imu_ready = True
        obj.source_is_fresh = MagicMock(return_value=True)
        obj.waypoint_format = 'local_xy'
        msg = SimpleNamespace(latitude=37., longitude=127., eastOffset=0.,
                              northOffset=0., status=1)

        def project(*args):
            # This is where the control thread could observe the old false flag.
            self.assertTrue(obj.gps_ready)
            return 10., 20.

        obj.proj_utm = project
        obj.gps_callback(msg)
        self.assertEqual((obj.current_x, obj.current_y), (10., 20.))
        imu = SimpleNamespace(orientation_covariance=[0.] * 9,
                              orientation=SimpleNamespace(x=0., y=0., z=0., w=1.))

        def quaternion(*args):
            self.assertTrue(obj.imu_ready)
            return 0., 0., 0.

        with patch('gps_waypoint_follower.euler_from_quaternion', side_effect=quaternion):
            obj.imu_callback(imu)
        msg.status = 0
        with patch('gps_waypoint_follower.rospy.logwarn_throttle'):
            obj.gps_callback(msg)
        self.assertFalse(obj.gps_ready)
        imu.orientation_covariance[0] = -1
        obj.imu_callback(imu)
        self.assertFalse(obj.imu_ready)

    def test_tracking_state_reaches_map_debug_and_expires(self):
        from gps_waypoint_follower import GPSWaypointFollower
        from map_debug import MapDebug
        obj = GPSWaypointFollower.__new__(GPSWaypointFollower)
        obj.current_waypoint_index = 0
        obj.route_sectors = [11]
        obj.tracking_state_pub = MagicMock()
        debug = MapDebug.__new__(MapDebug)
        debug.tracking = None
        for lane in (3, 2, 1):
            obj.active_raceline = lane
            obj.publish_tracking_state()
            message = obj.tracking_state_pub.publish.call_args[0][0]
            with patch('map_debug.time.monotonic', return_value=100.0):
                debug.on_tracking_state(message)
            state = debug.current_tracking(100.5)
            self.assertEqual({key: state[key] for key in ('valid', 'sector', 'raceline')},
                             {'valid': True, 'sector': 11, 'raceline': lane})
            self.assertIsNone(debug.current_tracking(101.0))
        obj.publish_tracking_state(valid=False)
        with patch('map_debug.time.monotonic', return_value=102.0):
            debug.on_tracking_state(obj.tracking_state_pub.publish.call_args[0][0])
        self.assertIsNone(debug.current_tracking(102.1))
        del obj.active_raceline
        obj.publish_tracking_state()
        self.assertEqual(json.loads(obj.tracking_state_pub.publish.call_args[0][0].data)['raceline'], 1)

    def follower(self):
        obj = RacelineTestFollower.__new__(RacelineTestFollower)
        obj.waypoint_file = str(ROOT / 'config/map_tunner_project.json')
        obj.waypoint_format = 'local_xy'
        obj.enable_local_avoidance = True
        with patch('raceline_test_follower.rospy.get_param',
                   side_effect=lambda _name, default=None: default):
            obj.load_waypoints()
        obj.waypoints = obj.raw_waypoints
        return obj

    def planner(self, obj, preferred_lane=3):
        return LocalAvoidancePlanner(obj.waypoints, obj.route_sectors,
                                     obj.project_lanes,
                                     lane_indices=obj.project_lane_indices,
                                     preferred_lane=preferred_lane)

    def test_elapsed_time_does_not_switch_global_route(self):
        obj = self.follower()
        obj.lookahead_speed_gain = 0.
        original = list(obj.waypoints)
        p = self.planner(obj)
        first = min(obj.project_lane_indices[2]) + 60
        for now in (0., 10., 20., 40., 100.):
            with patch('gps_waypoint_follower.time.monotonic', return_value=now):
                obj.update_dynamic_lookahead()
                d = p.plan(first, obj.waypoints[first], 5., [])
            self.assertEqual(d.state, 'LANE_CHANGING')
            self.assertEqual(p.active_lane, 3)
            self.assertEqual(obj.waypoints, original)
        self.assertFalse(hasattr(obj, 'sequence'))

    def test_saved_map_prefers_r3_then_r2_when_r3_is_blocked(self):
        obj = self.follower()
        first = min(obj.project_lane_indices[2]) + 60
        obstacle_index = first + 65
        x, y = obj.waypoints[obstacle_index]
        r1_obstacle = Obstacle(x, y, .8, kind='static')
        p = self.planner(obj)
        self.assertEqual(p.plan(first, obj.waypoints[first], 5., [r1_obstacle]).state,
                         'OVERTAKING')
        self.assertEqual(p.active_lane, 3)
        x3, y3 = obj.project_lane_indices[3][obstacle_index]
        p = self.planner(obj)
        decision = p.plan(first, obj.waypoints[first], 5., [
            r1_obstacle, Obstacle(x3, y3, .8, kind='static')])
        self.assertEqual(decision.state, 'OVERTAKING')
        self.assertEqual(p.active_lane, 2)
        self.assertTrue(decision.path)
        x2, y2 = obj.project_lane_indices[2][obstacle_index]
        p = self.planner(obj)
        blocked = p.plan(first, obj.waypoints[first], 5., [
            r1_obstacle, Obstacle(x2, y2, .8), Obstacle(x3, y3, .8)])
        self.assertEqual(blocked.state, 'TAILING')
        self.assertFalse(blocked.path)

    def test_lane_selection_is_reported_to_map_debug(self):
        obj = self.follower()
        obj.current_waypoint_index = min(obj.project_lane_indices[2]) + 60
        obj.tracking_state_pub = MagicMock()
        obj.local_planner = self.planner(obj)
        obj.local_planner.active_lane = 3
        obj.avoidance_state = 'OVERTAKING'
        obj.publish_tracking_state()
        state = json.loads(obj.tracking_state_pub.publish.call_args[0][0].data)
        self.assertEqual(state['raceline'], 3)
        self.assertEqual(state['local_lane'], 3)
        self.assertEqual(state['driving_state'], 'OVERTAKING')

    def test_saved_r3_returns_to_r3_after_passing_and_checks_rear_gap(self):
        obj = self.follower()
        p = self.planner(obj)
        first = min(obj.project_lane_indices[3])+60
        lead_index = first+65
        ego = obj.project_lane_indices[3][first]
        lead = Obstacle(*obj.project_lane_indices[3][lead_index], radius=.8, kind='static')
        passed = p.plan(first, ego, 5., [lead])
        self.assertEqual(passed.state, 'OVERTAKING')
        self.assertEqual(p.active_lane, 2)
        later = first+120
        returned = p.plan(later, obj.project_lane_indices[2][later], 5., [lead])
        self.assertEqual(returned.state, 'RETURNING')
        self.assertEqual(p.active_lane, 3)

    def test_saved_map_rear_velocity_blocks_r3_entry(self):
        obj = self.follower()
        p = self.planner(obj)
        first = min(obj.project_lane_indices[3])+60
        tangent = p._tangent(first)
        x, y = obj.project_lane_indices[3][first]
        rear = Obstacle(x-35.*tangent[0], y-35.*tangent[1], .8,
                        vx=18.*tangent[0], vy=18.*tangent[1], kind='dynamic')
        d = p.plan(first, obj.waypoints[first], 5., [rear])
        self.assertEqual(d.state, 'LANE_WAIT')
        self.assertIsNone(p.transition)
        self.assertIsNone(p.active_lane)

    def test_test_launch_default_prefers_r3(self):
        from xml.etree import ElementTree
        root = ElementTree.parse(ROOT.parent/'unita_launch/launch/test_map_raceline.launch').getroot()
        self.assertEqual(root.find("arg[@name='preferred_raceline']").attrib['default'], '3')

    def checkpoint_preferences(self):
        import yaml
        obj = self.follower()
        cp_xy = yaml.safe_load((ROOT/'config/map_tunner_checkpoints.yaml').read_text())['checkpoints']['11']['position']
        cp = dict(x=cp_xy[0], y=cp_xy[1])
        end = min(range(len(obj.waypoints)),
                  key=lambda i: math.hypot(obj.waypoints[i][0]-cp['x'], obj.waypoints[i][1]-cp['y']))
        # Deliberately keep the incompatible speed-profile CP11 index.
        obj.checkpoint_indices = {'11': 2586}
        params = {'~preferred_raceline': 3, '~preferred_raceline_sector': 11,
                  '~preferred_raceline_end_checkpoint': '11', '~preferred_raceline_after_checkpoint': 1}
        with patch('gps_waypoint_follower.rospy.get_param',
                   side_effect=lambda name, default=None: params.get(name, default)), \
                patch('gps_waypoint_follower.rospy.loginfo'):
            fallback, preferences = obj.local_lane_preferences()
        planner = LocalAvoidancePlanner(obj.waypoints, obj.route_sectors, obj.project_lanes,
                                       lane_indices=obj.project_lane_indices,
                                       preferred_lane=fallback, preferred_lane_indices=preferences,
                                       finish_line=obj.lane_preference_finish_line)
        return obj, planner, end

    def test_real_cp11_holds_r3_until_rear_clears_in_next_sector(self):
        obj, p, end = self.checkpoint_preferences()
        self.assertAlmostEqual(p.finish_line['y'], -217.855053961277)
        self.assertNotEqual(end, obj.checkpoint_indices['11'])
        for index in (end-30, end, end+1):
            ego = p._lane_point(3, obj.waypoints[index], index)
            d = p.plan(index, ego, 5., [], ego_yaw=-math.pi/2)
            self.assertEqual(d.state, 'RACELINE')
            self.assertEqual(p.effective_preferred_lane, 3)
            self.assertEqual(p.active_lane, 3)
            self.assertFalse(p.finish_line_passed)
        index = end+3
        d = p.plan(index, obj.waypoints[index], 5., [], ego_yaw=-math.pi/2)
        self.assertEqual(p.effective_preferred_lane, 1)
        self.assertTrue(p.finish_line_passed)
        self.assertIsNone(p.active_lane)

    def test_finish_line_return_still_waits_for_fast_rear_traffic(self):
        route = [(0., -float(i)) for i in range(180)]
        lanes = {n: [((n-1)*3.2, y) for _, y in route] for n in (2, 3)}
        line = dict(sector=11, lane=3, checkpoint='11', start_index=20,
                    end_index=80, y=-80., joins_primary=False)
        p = LocalAvoidancePlanner(route, [11]*180, lanes,
                                  lane_indices={n: dict(enumerate(pts)) for n, pts in lanes.items()},
                                  finish_line=line)
        p.plan(80, lanes[3][80], 5., [])
        rear = Obstacle(0., -53., .8, vy=-18., kind='dynamic')
        d = p.plan(83, lanes[3][83], 5., [rear])
        self.assertEqual(d.state, 'LANE_WAIT')
        self.assertEqual(p.active_lane, 3)
        self.assertEqual(p.effective_preferred_lane, 1)
        d = p.plan(83, lanes[3][83], 5., [])
        self.assertEqual(d.state, 'RETURNING')
        self.assertIsNone(p.active_lane)

    def test_offset_lane_nearest_r1_index_does_not_end_priority_before_crossing(self):
        # R1's nearest sample lies ahead of this car on a slanted R3.
        route = [(0.3*i, -float(i)) for i in range(160)]
        lanes = {n: [(x+(n-1)*3.2, y) for x, y in route] for n in (2, 3)}
        mapping = {n: dict(enumerate(points)) for n, points in lanes.items()}
        end = 60
        planner = LocalAvoidancePlanner(route, [11]*len(route), lanes,
                                       lane_indices=mapping,
                                       preferred_lane_indices={i: 3 for i in range(20, end+1)})
        obj = RacelineTestFollower.__new__(RacelineTestFollower)
        obj.waypoints = route
        obj.loop_path = False
        obj.current_waypoint_index = end-5
        tangent = planner._tangent(end)
        cp = lanes[3][end]
        for offset, expected in ((-.2, 3), (0., 3), (.2, 1)):
            ego = (cp[0]+offset*tangent[0], cp[1]+offset*tangent[1])
            obj.current_x, obj.current_y = ego
            obj.first_nearest_search = True
            index = obj.find_nearest_waypoint()
            self.assertGreater(index, end)
            decision = planner.plan(index, ego, 5., [])
            self.assertEqual(planner.effective_preferred_lane, expected)
            self.assertEqual(decision.state, 'RETURNING' if expected == 1 else 'RACELINE')

    def test_cp11_crossing_uses_rearmost_y_and_is_independent_of_x(self):
        obj, p, end = self.checkpoint_preferences()
        y = p.finish_line['y']
        for index in (end-1, end, end+1):
            for x in (-1000., 0., 1000.):
                for delta, expected in ((.2, 3), (0., 3), (-.79, 3), (-.791, 1)):
                    with self.subTest(index=index, x=x, delta=delta):
                        self.assertEqual(p._requested_lane(index, (x, y+delta), -math.pi/2), expected)
        # At a yaw angle, the trailing corner crosses later than rear center.
        yaw = math.radians(-80.)
        rear_offset = -p.rear_extent*math.sin(yaw)+p.half_width*abs(math.cos(yaw))
        self.assertEqual(p._requested_lane(end, (75., y-rear_offset+.001), yaw), 3)
        self.assertEqual(p._requested_lane(end, (75., y-rear_offset-.001), yaw), 1)

    def test_entire_saved_r3_and_manual_extension_keep_priority(self):
        obj, p, end = self.checkpoint_preferences()
        for index in range(p.finish_line['start_index'], end+1):
            ego = obj.project_lane_indices[3][index]
            decision = p.plan(index, ego, 5., [], ego_yaw=-math.pi/2)
            self.assertEqual(p.effective_preferred_lane, 3, (index, ego, decision))
            self.assertEqual(p.active_lane, 3, (index, ego, decision))
        self.assertEqual(p._requested_lane(end+50, obj.waypoints[end+50]), 1)
        self.assertEqual(p._requested_lane(100, obj.waypoints[100]), 1)
        # A new lap returns to R3 at the split.
        start = p.finish_line['start_index']
        self.assertEqual(p._requested_lane(start, obj.project_lane_indices[3][start]), 3)

    def test_obstacle_avoidance_in_manual_extension_keeps_r3_priority(self):
        obj, p, end = self.checkpoint_preferences()
        index = min(obj.project_lane_indices[3], key=lambda i: abs(obj.waypoints[i][1]+150.))
        obstacle = Obstacle(*obj.project_lane_indices[3][index+50], radius=.8)
        decision = p.plan(index, obj.project_lane_indices[3][index], 5., [obstacle])
        self.assertEqual(p.effective_preferred_lane, 3)
        self.assertIn(decision.state, ('OVERTAKING', 'TAILING'))

    def test_missing_extension_stops_without_redefining_r3_priority(self):
        obj, p, end = self.checkpoint_preferences()
        index = end-60
        del p.lane_indices[3][index]
        decision = p.plan(index, (75., obj.waypoints[index][1]), 5., [])
        self.assertEqual(p.effective_preferred_lane, 3)
        self.assertEqual(decision.state, 'BLOCKED')
        self.assertEqual(decision.speed_cap_kmh, 0.)

    def test_priority_mapping_does_not_cover_other_sectors_or_after_cp11(self):
        obj, p, end = self.checkpoint_preferences()
        self.assertEqual(p.preferred_lane_indices[end], 3)
        self.assertNotIn(end+1, p.preferred_lane_indices)
        for i, sector in enumerate(obj.route_sectors):
            if sector != 11:
                self.assertNotIn(i, p.preferred_lane_indices)

    def test_lane_ends_cannot_be_extrapolated_for_passing(self):
        obj = self.follower()
        p = self.planner(obj)
        first = max(obj.project_lane_indices[3]) - 10
        x, y = obj.waypoints[first+5]
        decision = p.plan(first, obj.waypoints[first], 5., [Obstacle(x, y, .8)])
        self.assertEqual(decision.state, 'TAILING')
        self.assertFalse(decision.path)

    def test_follower_controls_using_selected_local_path_and_stops_on_stale_input(self):
        import rospy
        obj = self.follower()
        first = min(obj.project_lane_indices[2]) + 60
        obj.current_waypoint_index = first
        obj.current_x, obj.current_y = obj.waypoints[first]
        next_x, next_y = obj.waypoints[first+1]
        obj.current_yaw = math.atan2(next_y-obj.current_y, next_x-obj.current_x)
        obj.loop_path = True
        obj.path_ready = obj.gps_ready = obj.imu_ready = True
        obj.drive_mode = None
        obj.gps_received = obj.imu_received = 200.
        obj.pose_sensor_timeout = 1.
        obj.first_nearest_search = False
        obj.last_tracking_position = None
        obj.position_reset_distance = 10.
        obj.goal_tolerance = 2.
        obj.lookahead_distance = 5.
        obj.lookahead_speed_gain = 0.
        obj.current_speed_kmh = 18.
        obj.wheelbase = 3.
        obj.steering_sign = 1.
        obj.max_steering_rad = .5
        obj.longitudinal_control_mode = 'velocity'
        obj.enable_velocity_planning = False
        obj.target_speed_kmh = 20.
        obj.path_curvatures = []
        obj.guard_limit_topic = ''
        obj.local_planner = self.planner(obj)
        obj.obstacle_stamp = 100.
        obj.obstacle_received = 200.
        obj.obstacles = [Obstacle(*obj.waypoints[first+65], radius=.8),
                         Obstacle(*obj.project_lane_indices[2][first+65], radius=.8)]
        for attr in ('local_path_pub', 'tracking_state_pub', 'publish_control',
                     'publish_full_stop', 'target_speed_pub', 'current_speed_pub',
                     'speed_zone_pub'):
            setattr(obj, attr, MagicMock())
        with patch('gps_waypoint_follower.time.monotonic', return_value=200.), \
                patch('gps_waypoint_follower.rospy.Time.now', return_value=rospy.Time(100)), \
                patch('gps_waypoint_follower.rospy.loginfo_throttle'), \
                patch('gps_waypoint_follower.rospy.logwarn_throttle'):
            obj.control_loop(None)
            self.assertEqual(obj.local_planner.active_lane, 3)
            path = obj.local_path_pub.publish.call_args[0][0]
            self.assertTrue(path.poses)
            target = obj.local_target([(p.pose.position.x, p.pose.position.y)
                                       for p in path.poses])
            expected_steering = obj.calculate_steering(target[1], target[2], target[3])
            obj.publish_control.assert_called_once_with(expected_steering, 20.)
            obj.publish_full_stop.assert_not_called()
            obj.obstacle_received = 199.
            obj.control_loop(None)
            obj.publish_full_stop.assert_called_once()
            self.assertEqual(obj.avoidance_state, 'SENSOR_HOLD')


if __name__ == '__main__':
    unittest.main()
