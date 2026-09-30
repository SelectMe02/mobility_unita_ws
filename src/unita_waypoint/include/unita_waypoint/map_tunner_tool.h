#pragma once

#include <ros/ros.h>
#include <rviz/tool.h>
#include <geometry_msgs/PointStamped.h>

namespace unita_waypoint {

class MapTunnerTool : public rviz::Tool {
  Q_OBJECT

 public:
  MapTunnerTool();
  void onInitialize() override;
  void activate() override;
  void deactivate() override;
  int processMouseEvent(rviz::ViewportMouseEvent& event) override;

 private:
  ros::NodeHandle node_;
  ros::Publisher click_pub_;
  ros::Publisher smooth_pub_;
};

}  // namespace unita_waypoint
