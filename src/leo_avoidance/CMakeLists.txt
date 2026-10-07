cmake_minimum_required(VERSION 3.0.2)
project(leo_avoidance)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs std_msgs morai_msgs geometry_msgs visualization_msgs)
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS scripts/blackout_udp_control.py
  scripts/ego_odometry_node.py scripts/driving_state_node.py
  scripts/ftg_visualization_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config scenarios DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_ftg.py)
  catkin_add_nosetests(test/test_blackout_watchdog.py)
  catkin_add_nosetests(test/test_scan_odometry.py)
endif()
