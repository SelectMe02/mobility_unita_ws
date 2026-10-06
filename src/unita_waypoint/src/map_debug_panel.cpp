#include <unita_waypoint/map_debug_panel.h>

#include <pluginlib/class_list_macros.h>
#include <QGridLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QTimer>
#include <QVBoxLayout>

namespace unita_waypoint {

MapDebugPanel::MapDebugPanel(QWidget* parent) : rviz::Panel(parent) {
  auto* layout = new QVBoxLayout(this);
  auto* title = new QLabel(QString::fromUtf8("K-City 주행 상태"));
  QFont title_font = title->font();
  title_font.setBold(true);
  title_font.setPointSize(title_font.pointSize() + 3);
  title->setFont(title_font);
  layout->addWidget(title);
  auto* hint = new QLabel(QString::fromUtf8("노란 화살표는 차량 위치와 진행 방향입니다.\n"
                                            "하늘색 선은 저장된 Raceline 1입니다."));
  hint->setWordWrap(true);
  layout->addWidget(hint);
  auto* grid = new QGridLayout();
  speed_ = new QLabel(QString::fromUtf8("대기 중"));
  position_ = new QLabel(QString::fromUtf8("대기 중"));
  heading_ = new QLabel(QString::fromUtf8("대기 중"));
  sector_ = new QLabel(QString::fromUtf8("대기 중"));
  raceline_ = new QLabel(QString::fromUtf8("대기 중"));
  driving_state_ = new QLabel(QString::fromUtf8("대기 중"));
  obstacle_ = new QLabel(QString::fromUtf8("없음"));
  finish_ = new QLabel(QString::fromUtf8("대기 중"));
  const QString names[] = {QString::fromUtf8("현재 속도"), QString::fromUtf8("차량 위치"),
                           QString::fromUtf8("헤딩"), QString::fromUtf8("현재 Sector"),
                           QString::fromUtf8("추종 Raceline"),
                           QString::fromUtf8("주행 상태"), QString::fromUtf8("전방 장애물"),
                           QString::fromUtf8("CP11 끝선")};
  QLabel* values[] = {speed_, position_, heading_, sector_, raceline_, driving_state_, obstacle_, finish_};
  for (int row = 0; row < 8; ++row) {
    auto* name = new QLabel(names[row]);
    QFont font = name->font();
    font.setBold(true);
    name->setFont(font);
    grid->addWidget(name, row, 0);
    grid->addWidget(values[row], row, 1);
  }
  layout->addLayout(grid);
  layout->addStretch();
  subscriber_ = node_.subscribe("/map_debug/status", 1, &MapDebugPanel::onStatus, this);
  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &MapDebugPanel::refresh);
  timer_->start(100);
}

void MapDebugPanel::onStatus(const std_msgs::String::ConstPtr& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  status_ = message->data;
  status_received_ = ros::WallTime::now();
}

void MapDebugPanel::refresh() {
  std::string status;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!status_received_.isZero() &&
        (ros::WallTime::now() - status_received_).toSec() < 1.0)
      status = status_;
  }
  const auto document = QJsonDocument::fromJson(QByteArray::fromStdString(status));
  const auto data = document.object();
  const auto waiting = QString::fromUtf8("대기 중");
  const bool available = data.value("available").toBool(false);
  const auto speed = data.value("speed_kmh");
  speed_->setText(speed.isDouble() ? QString::number(speed.toDouble(), 'f', 1) + " km/h" : waiting);
  position_->setText(available ? QString("X %1 m, Y %2 m")
      .arg(data.value("x").toDouble(), 0, 'f', 2)
      .arg(data.value("y").toDouble(), 0, 'f', 2) : waiting);
  heading_->setText(available ? QString::number(data.value("heading_deg").toDouble(), 'f', 1)
      + QString::fromUtf8("°") : waiting);
  sector_->setText(available ? QString("S%1 / 15").arg(data.value("sector").toInt()) : waiting);
  const auto raceline = data.value("raceline");
  const auto local_lane = data.value("local_lane");
  raceline_->setToolTip(data.value("project_file").toString());
  raceline_->setText(available && local_lane.isDouble()
      ? QString("Local Path / Raceline %1").arg(local_lane.toInt())
      : available && raceline.isDouble()
      ? QString("Raceline %1").arg(raceline.toInt()) : waiting);
  const auto current_lane = data.value("current_raceline");
  const auto preferred_lane = data.value("preferred_raceline");
  const auto finish_y = data.value("finish_line_y");
  const auto rear_y = data.value("finish_line_rear_y");
  const auto passed = data.value("finish_line_passed");
  finish_->setText(available && finish_y.isDouble() && rear_y.isDouble() && passed.isBool()
      ? QString::fromUtf8("%1 / 뒤쪽 Y %2 / 끝선 Y %3")
          .arg(passed.toBool() ? QString::fromUtf8("후미 통과") : QString::fromUtf8("후미 통과 전"))
          .arg(rear_y.toDouble(), 0, 'f', 2).arg(finish_y.toDouble(), 0, 'f', 2)
      : waiting);
  if (available && current_lane.isDouble() && preferred_lane.isDouble()) {
    const int target = local_lane.isDouble() ? local_lane.toInt()
        : raceline.isDouble() ? raceline.toInt() : current_lane.toInt();
    raceline_->setText(current_lane.toInt() == target
        ? QString::fromUtf8("현재 R%1 / 우선 R%2").arg(current_lane.toInt()).arg(preferred_lane.toInt())
        : QString::fromUtf8("현재 R%1 → 목표 R%2 / 우선 R%3")
            .arg(current_lane.toInt()).arg(target).arg(preferred_lane.toInt()));
  }
  const auto state = data.value("driving_state").toString();
  QString state_text = waiting;
  if (available && !state.isEmpty()) {
    if (state == "RACELINE") state_text = QString::fromUtf8("경로 추종");
    else if (state == "TAILING") state_text = QString::fromUtf8("뒤따르기 (TAILING)");
    else if (state == "OVERTAKING") state_text = QString::fromUtf8("추월 (OVERTAKING)");
    else if (state == "RETURNING") state_text = QString::fromUtf8("우선 차선 복귀 (RETURNING)");
    else if (state == "LANE_CHANGING") state_text = QString::fromUtf8("차선 변경 (LANE_CHANGING)");
    else if (state == "LANE_WAIT") state_text = QString::fromUtf8("차선 변경 대기 (LANE_WAIT)");
    else if (state == "SENSOR_HOLD") state_text = QString::fromUtf8("장애물 센서 대기");
    else if (state == "BLOCKED") state_text = QString::fromUtf8("회피 경로 차단");
    else state_text = state;
  }
  const auto reason = data.value("lane_change_reason").toString();
  if (available && reason == "rear_or_crossed_lane_busy")
    state_text += QString::fromUtf8(" / 후방·측방 차량");
  else if (available && reason == "slowing_for_lane_change")
    state_text += QString::fromUtf8(" / 변경 전 감속");
  driving_state_->setText(state_text);
  const auto obstacle_kind = data.value("obstacle_kind").toString();
  const auto obstacle_distance = data.value("obstacle_distance_m");
  obstacle_->setText(available && obstacle_distance.isDouble()
      ? QString::fromUtf8("%1 / %2 m").arg(obstacle_kind == "dynamic"
          ? QString::fromUtf8("동적") : obstacle_kind == "static"
          ? QString::fromUtf8("정적") : QString::fromUtf8("분류 중"))
          .arg(obstacle_distance.toDouble(), 0, 'f', 1)
      : QString::fromUtf8("없음"));
}

}  // namespace unita_waypoint

PLUGINLIB_EXPORT_CLASS(unita_waypoint::MapDebugPanel, rviz::Panel)
