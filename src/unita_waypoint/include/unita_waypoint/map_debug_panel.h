#pragma once

#include <mutex>
#include <string>

#include <ros/ros.h>
#include <rviz/panel.h>
#include <std_msgs/String.h>

class QLabel;
class QTimer;

namespace unita_waypoint {

class MapDebugPanel : public rviz::Panel {
  Q_OBJECT
 public:
  explicit MapDebugPanel(QWidget* parent = nullptr);

 private Q_SLOTS:
  void refresh();

 private:
  void onStatus(const std_msgs::String::ConstPtr& message);
  ros::NodeHandle node_;
  ros::Subscriber subscriber_;
  std::mutex mutex_;
  std::string status_;
  ros::WallTime status_received_;
  QLabel* speed_;
  QLabel* position_;
  QLabel* heading_;
  QLabel* sector_;
  QLabel* raceline_;
  QLabel* driving_state_;
  QLabel* obstacle_;
  QLabel* finish_;
  QTimer* timer_;
};

}  // namespace unita_waypoint
