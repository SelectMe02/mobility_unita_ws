#!/usr/bin/env python3
"""Saved raceline geometry with obstacle-driven local route selection."""
import sys
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import rospkg
import rospy

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from gps_waypoint_follower import GPSWaypointFollower
from driving_log_session import announced_directory, new_run_directory


class RacelineTestFollower(GPSWaypointFollower):
    def __init__(self):
        self.diagnostic_file = None
        self.diagnostic_pending = None
        self.last_diagnostic_key = None
        self.last_diagnostic_time = 0.0
        super().__init__()
        if rospy.get_param('~diagnostic_log', False):
            # The recorder names the run folder (log/run_YYYYmmdd_HHMMSS);
            # buffer records until it announces this launch's folder.
            self.diagnostic_pending = dict(
                explicit=rospy.get_param('~diagnostic_directory', ''),
                session=rospy.get_param('~diagnostic_session', ''),
                root=rospy.get_param('~diagnostic_root',
                    str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'log')),
                deadline=time.monotonic()+10., buffer=[])
            self._diagnostic('start', force=True)

    def _open_diagnostic_file(self):
        pending = self.diagnostic_pending
        if pending is None:
            return False
        if pending['explicit']:
            directory = Path(pending['explicit'])
            directory.mkdir(parents=True, exist_ok=True)
        else:
            directory = announced_directory(rospy, pending['session'])
            if directory is None:
                if time.monotonic() < pending['deadline']:
                    return False
                directory = new_run_directory(pending['root'])
                rospy.logwarn('Driving log recorder did not announce a folder; using %s', directory)
        filename = 'raceline_%s_%d.jsonl' % (datetime.now().strftime('%Y%m%d_%H%M%S'), os.getpid())
        self.diagnostic_file = (directory / filename).open('a', buffering=1)
        rospy.on_shutdown(self.diagnostic_file.close)
        rospy.loginfo('Raceline diagnostic log: %s', self.diagnostic_file.name)
        for line in pending['buffer']:
            self.diagnostic_file.write(line)
        self.diagnostic_pending = None
        return True

    def _diagnostic(self, event, force=False, **extra):
        if self.diagnostic_file is None and not self._open_diagnostic_file():
            if self.diagnostic_pending is None:
                return
        planner = getattr(self, 'local_planner', None)
        record = dict(
            time=datetime.now().astimezone().isoformat(timespec='milliseconds'),
            ros_time=rospy.Time.now().to_sec(),
            event=event,
            waypoint_index=getattr(self, 'current_waypoint_index', None),
            sector=(self.route_sectors[self.current_waypoint_index]
                    if getattr(self, 'route_sectors', None)
                    and 0 <= self.current_waypoint_index < len(self.route_sectors) else None),
            x=getattr(self, 'current_x', None), y=getattr(self, 'current_y', None),
            speed_kmh=getattr(self, 'current_speed_kmh', None),
            state=getattr(self, 'avoidance_state', None),
            lane=getattr(planner, 'active_lane', None),
            current_lane=getattr(planner, 'current_lane', None),
            preferred_lane=getattr(planner, 'effective_preferred_lane', None),
            preferred_reason=getattr(planner, 'preference_reason', None),
            lane_change_reason=getattr(self, 'lane_change_reason', None),
            obstacle_kind=getattr(self, 'avoidance_kind', None),
            obstacle_distance_m=getattr(self, 'avoidance_distance', None),
            obstacle_count=len(getattr(self, 'obstacles', [])),
            obstacle_age_s=(time.monotonic() - self.obstacle_received
                            if getattr(self, 'obstacle_received', None) is not None else None),
            yaw_deg=(math.degrees(self.current_yaw)
                     if isinstance(getattr(self, 'current_yaw', None), float)
                     and math.isfinite(self.current_yaw) else None),
            transition=getattr(planner, 'transition', None),
            lane_hold=getattr(planner, 'lane_hold', None),
            **extra)
        key = (event, record['state'], record['lane'], record['preferred_lane'],
               record['preferred_reason'], record['lane_change_reason'],
               record.get('stop_reason'))
        now = time.monotonic()
        if force or key != self.last_diagnostic_key or now - self.last_diagnostic_time >= 1.0:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
            if self.diagnostic_file is not None:
                self.diagnostic_file.write(line)
            elif len(self.diagnostic_pending['buffer']) < 5000:
                self.diagnostic_pending['buffer'].append(line)
            self.last_diagnostic_key = key
            self.last_diagnostic_time = now

    def publish_full_stop(self):
        self._diagnostic('stop', stop_reason=getattr(self, 'stop_reason', 'unknown'))
        super().publish_full_stop()

    def publish_control(self, steering, velocity):
        super().publish_control(steering, velocity)
        held = getattr(self, 'pedal_reason', '') == 'clearance_confirmation'
        self._diagnostic('holding' if held else 'drive', target_speed_kmh=0. if held else velocity,
                         requested_target_speed_kmh=velocity,
                         stop_reason='clearance_confirmation' if held else None, steering_rad=steering,
                         speed_limits=getattr(self, 'diagnostic_speed_limits', {}))

    def load_waypoints(self):
        if Path(self.waypoint_file).suffix.lower() != '.json':
            raise ValueError('raceline selection requires a Map Tuner JSON project')
        super().load_waypoints()


if __name__ == '__main__':
    try:
        follower = RacelineTestFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
