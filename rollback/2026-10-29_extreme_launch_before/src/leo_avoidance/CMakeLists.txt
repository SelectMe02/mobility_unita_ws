cmake_minimum_required(VERSION 3.0.2)
project(leo_avoidance)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs std_msgs morai_msgs geometry_msgs visualization_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS scripts/blackout_udp_control.py
  scripts/ego_odometry_node.py scripts/driving_state_node.py
  scripts/ftg_visualization_node.py scripts/route_avoidance_node.py
  scripts/blackout_udp_control_local.py scripts/driving_state_node_local.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config scenarios DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_ftg.py)
  catkin_add_nosetests(test/test_blackout_watchdog.py)
  catkin_add_nosetests(test/test_scan_odometry.py)
  catkin_add_nosetests(test/test_ftg_local.py)
  catkin_add_nosetests(test/test_blackout_watchdog_local.py)
  catkin_add_nosetests(test/test_scan_odometry_local.py)
  catkin_add_nosetests(test/test_rejoin_route_local.py)
  catkin_add_nosetests(test/test_obstacle_tracker.py)
  catkin_add_nosetests(test/test_route_pose_sync.py)
  catkin_add_nosetests(test/test_route_sector_lane_filter.py)
  catkin_add_nosetests(test/test_route_avoidance.py)
endif()
