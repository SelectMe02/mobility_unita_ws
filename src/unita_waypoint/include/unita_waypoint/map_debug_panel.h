#pragma once

#include <mutex>
#include <string>
#include <vector>
#include <QString>
#include <QImage>

#include <ros/ros.h>
#include <rviz/panel.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/Image.h>

class QLabel;
class QTimer;
class QLineEdit;
class QPushButton;

namespace rviz {
class RenderPanel;
class VisualizationManager;
}

namespace unita_waypoint {

class MapDebugPanel : public rviz::Panel {
  Q_OBJECT
 public:
  explicit MapDebugPanel(QWidget* parent = nullptr);
  ~MapDebugPanel() override;
  QSize sizeHint() const override;

 private Q_SLOTS:
  void refresh();
  void resetFollowCamera();

 private:
  void onStatus(const std_msgs::String::ConstPtr& message);
  void onRearImage(const sensor_msgs::Image::ConstPtr& message);
  void onGuard(const std_msgs::String::ConstPtr& message);
  void onSignalCap(const std_msgs::Float64::ConstPtr& message);
  void refreshCamerasAndSignal();
  void applyParameter(size_t index);
  void initializeLidarView();
  struct ParameterRow {
    QString name;
    QLabel* label;
    QLabel* current;
    QLineEdit* edit;
    QPushButton* apply;
  };
  ros::NodeHandle node_;
  ros::Subscriber subscriber_;
  std::vector<ros::Subscriber> camera_subscribers_;
  ros::Publisher parameter_pub_;
  ros::Publisher perception_view_pub_;
  std::mutex mutex_;
  std::string status_;
  ros::WallTime status_received_;
  sensor_msgs::Image::ConstPtr rear_image_;
  sensor_msgs::Image::ConstPtr rendered_rear_;
  QImage rear_frame_;
  ros::WallTime rear_received_, guard_received_, cap_received_;
  std::string guard_;
  double signal_cap_kmh_ = 0.0;
  QLabel* position_;
  QLabel* sector_;
  QLabel* raceline_;
  QLabel* raceline_width_;
  QLabel* parameter_message_;
  QLabel* rear_view_;
  QLabel* signal_light_;
  QLabel* signal_action_;
  QLabel* signal_metrics_;
  QLabel* signal_reason_;
  QLabel* vehicle_dimensions_;
  rviz::RenderPanel* lidar_panel_;
  rviz::VisualizationManager* lidar_manager_;
  std::vector<ParameterRow> parameters_;
  QTimer* timer_;
};

}  // namespace unita_waypoint
