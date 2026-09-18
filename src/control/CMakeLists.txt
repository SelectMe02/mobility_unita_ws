cmake_minimum_required(VERSION 3.0.2)
project(control)
find_package(catkin REQUIRED COMPONENTS geometry_msgs morai_msgs nav_msgs rospy sensor_msgs std_msgs std_srvs)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS geometry_msgs morai_msgs nav_msgs rospy sensor_msgs std_msgs std_srvs)
catkin_install_python(PROGRAMS scripts/morai_udp_control.py scripts/competition_udp_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".gitkeep" EXCLUDE)
install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_udp_protocol.py)
  catkin_add_nosetests(test/test_competition_protocol.py)
endif()
