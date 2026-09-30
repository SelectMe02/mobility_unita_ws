#pragma once

#include <ros/ros.h>
#include <rviz/panel.h>
#include <std_msgs/String.h>

#include <QString>

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;

namespace unita_waypoint {

class MapTunnerPanel : public rviz::Panel {
  Q_OBJECT

 public:
  explicit MapTunnerPanel(QWidget* parent = nullptr);
  void onInitialize() override;

 private Q_SLOTS:
  void updateState(const QString& json);

 private:
  void stateCallback(const std_msgs::String::ConstPtr& message);
  void command(const std::string& value);
  void activateEditTool();
  void activateInteractTool();
  void applySpeed();

  ros::NodeHandle node_;
  ros::Publisher command_pub_;
  ros::Subscriber state_sub_;
  QComboBox* sector_box_;
  QComboBox* lane_box_;
  QLabel* sector_label_;
  QLabel* lane_label_;
  QLabel* point_label_;
  QLabel* status_label_;
  QLineEdit* speed_edit_;
  QPushButton* add_lane_button_;
  QPushButton* remove_lane_button_;
  QPushButton* add_point_button_;
  QPushButton* select_point_button_;
  QPushButton* interpolate_button_;
  QPushButton* smooth_button_;
  QPushButton* resize_button_;
  QPushButton* speed_apply_button_;
  QPushButton* delete_point_button_;
  QPushButton* save_button_;
};

}  // namespace unita_waypoint
