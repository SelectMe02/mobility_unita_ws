#!/usr/bin/env python3
"""Saved raceline geometry with obstacle-driven local route selection."""
import sys
from pathlib import Path

import rospkg
import rospy

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('unita_waypoint')) / 'scripts'))
from gps_waypoint_follower import GPSWaypointFollower


class RacelineTestFollower(GPSWaypointFollower):
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
