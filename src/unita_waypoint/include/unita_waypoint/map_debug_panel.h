#pragma once

#include <mutex>
#include <string>
#include <vector>
#include <QString>

#include <ros/ros.h>
#include <rviz/panel.h>
#include <std_msgs/String.h>

class QLabel;
class QTimer;
class QLineEdit;
class QPushButton;

namespace unita_waypoint {

class MapDebugPanel : public rviz::Panel {
  Q_OBJECT
 public:
  explicit MapDebugPanel(QWidget* parent = nullptr);

 private Q_SLOTS:
  void refresh();

 private:
  void onStatus(const std_msgs::String::ConstPtr& message);
  void applyParameter(size_t index);
  struct ParameterRow {
    QString name;
    QLabel* label;
    QLabel* current;
    QLineEdit* edit;
    QPushButton* apply;
  };
  ros::NodeHandle node_;
  ros::Subscriber subscriber_;
  ros::Publisher parameter_pub_;
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
  QLabel* raceline_width_;
  QLabel* parameter_message_;
  QLabel* ground_status_;
  std::vector<ParameterRow> parameters_;
  QTimer* timer_;
};

}  // namespace unita_waypoint
