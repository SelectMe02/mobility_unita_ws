cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)
catkin_package(CATKIN_DEPENDS geometry_msgs rospy sensor_msgs std_msgs)

# Register executable ROS nodes with catkin_install_python when added.
# Add catkin_python_setup() when src/ contains an installable Python package.
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".gitkeep" EXCLUDE)
install(FILES README.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
