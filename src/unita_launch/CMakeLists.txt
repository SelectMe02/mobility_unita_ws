cmake_minimum_required(VERSION 3.0.2)
project(unita_launch)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  std_msgs
  visualization_msgs
)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS rospy geometry_msgs nav_msgs std_msgs visualization_msgs)

catkin_install_python(PROGRAMS scripts/waypoint_visualizer.py scripts/generate_start_spawn.py scripts/field_check.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".gitkeep" EXCLUDE)
install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_path_metrics.py)
  catkin_add_nosetests(test/test_spawn_config.py)
  catkin_add_nosetests(test/test_field_diagnostics.py)
endif()
