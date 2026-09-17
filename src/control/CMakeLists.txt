cmake_minimum_required(VERSION 3.0.2)
project(control)
find_package(catkin REQUIRED COMPONENTS geometry_msgs morai_msgs nav_msgs rospy sensor_msgs std_msgs)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS geometry_msgs morai_msgs nav_msgs rospy sensor_msgs std_msgs)
catkin_install_python(PROGRAMS scripts/morai_udp_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".gitkeep" EXCLUDE)
install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_udp_protocol.py)
endif()
