#include <unita_waypoint/map_debug_panel.h>

#include <pluginlib/class_list_macros.h>
#include <QGridLayout>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <cmath>

namespace unita_waypoint {

MapDebugPanel::MapDebugPanel(QWidget* parent) : rviz::Panel(parent) {
  setMinimumWidth(480);
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
  raceline_width_ = new QLabel(QString::fromUtf8("대기 중"));
  const QString names[] = {QString::fromUtf8("현재 속도"), QString::fromUtf8("차량 위치"),
                           QString::fromUtf8("헤딩"), QString::fromUtf8("현재 Sector"),
                           QString::fromUtf8("추종 Raceline"),
                           QString::fromUtf8("주행 상태"), QString::fromUtf8("전방 장애물"),
                           QString::fromUtf8("Raceline 폭")};
  QLabel* values[] = {speed_, position_, heading_, sector_, raceline_, driving_state_, obstacle_, raceline_width_};
  for (int row = 0; row < 8; ++row) {
    auto* name = new QLabel(names[row]);
    QFont font = name->font();
    font.setBold(true);
    name->setFont(font);
    grid->addWidget(name, row, 0);
    grid->addWidget(values[row], row, 1);
  }
  layout->addLayout(grid);
  ground_status_ = new QLabel(QString::fromUtf8("노면 필터: 주행 노드 대기"));
  ground_status_->setWordWrap(true);
  layout->addWidget(ground_status_);
  auto* parameter_title = new QLabel(QString::fromUtf8("실시간 장애물·회피 설정"));
  parameter_title->setStyleSheet("font-weight: bold;");
  layout->addWidget(parameter_title);
  auto* parameter_hint = new QLabel(QString::fromUtf8(
      "값 입력 후 Enter 또는 적용을 누르세요. 주행 노드가 적용한 값을 표시합니다.\n"
      "보라색 영역은 현재 경로를 따라 잰 obstacle_horizon 범위입니다."));
  parameter_hint->setWordWrap(true);
  layout->addWidget(parameter_hint);
  const QString keys[] = {"cluster_min_size_m", "cluster_max_size_m",
      "front_range_m", "rear_range_m", "side_range_m",
      "dynamic_speed_threshold_mps", "obstacle_min_height_m", "obstacle_max_height_m",
      "ground_plane_tolerance_m", "obstacle_min_plane_height_m", "lane_edge_band_m", "imu_max_skew_s",
      "ground_max_slope_deg", "ground_reference_z_m", "track_confirm_time_s",
      "obstacle_horizon_m", "avoidance_margin_m", "lane_change_gap_m",
      "lane_change_lateral_accel_mps2", "lane_change_max_heading_deg", "avoidance_reaction_s", "avoidance_decel_mps2",
      "avoidance_brake_limit", "avoidance_clear_hold_s",
      "restart_accel_rate_per_s", "raceline_width",
      "sector11_min_speed_kmh", "lane_change_speed_boost_kmh",
      "lane_change_cancel_scans", "lane_change_cancel_max_progress",
      "route_max_error_m", "blackout_route_max_error_m",
      "pass_speed_kmh", "arc_lane_change_max_kmh", "route_pass_speed_kmh",
      "ftg_no_return_speed_mps", "ftg_wall_min_clearance_speed_mps",
      "ftg_near_obstacle_speed_mps", "ftg_wall_follow_min_speed_mps"};
  const int parameter_count = sizeof(keys) / sizeof(keys[0]);
  auto* table = new QTableWidget(parameter_count, 4, this);
  table->setHorizontalHeaderLabels({QString::fromUtf8("파라미터 / 설명"),
      QString::fromUtf8("현재값"), QString::fromUtf8("변경값"), QString::fromUtf8("적용")});
  table->verticalHeader()->hide();
  table->setSelectionMode(QAbstractItemView::NoSelection);
  table->setMinimumHeight(250);
  table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
  for (int column = 1; column < 4; ++column)
    table->horizontalHeader()->setSectionResizeMode(column, QHeaderView::ResizeToContents);
  for (int row = 0; row < parameter_count; ++row) {
    ParameterRow entry{keys[row], new QLabel(keys[row]),
        new QLabel(QString::fromUtf8("대기 중")), new QLineEdit(),
        new QPushButton(QString::fromUtf8("적용"))};
    entry.label->setWordWrap(true);
    entry.current->setWordWrap(true);
    entry.edit->setFixedWidth(75);
    entry.edit->setPlaceholderText(QString::fromUtf8("숫자"));
    if (keys[row] == "raceline_width") {
      entry.edit->setReadOnly(true);
      entry.apply->setEnabled(false);
      entry.apply->setText(QString::fromUtf8("자동"));
    }
    table->setCellWidget(row, 0, entry.label);
    table->setCellWidget(row, 1, entry.current);
    table->setCellWidget(row, 2, entry.edit);
    table->setCellWidget(row, 3, entry.apply);
    table->setRowHeight(row, 68);
    parameters_.push_back(entry);
    connect(entry.apply, &QPushButton::clicked, this, [this, row]() { applyParameter(row); });
    connect(entry.edit, &QLineEdit::returnPressed, this, [this, row]() { applyParameter(row); });
  }
  layout->addWidget(table, 1);
  parameter_message_ = new QLabel(QString::fromUtf8("주행 노드가 연결되면 적용값을 확인할 수 있습니다."));
  parameter_message_->setWordWrap(true);
  layout->addWidget(parameter_message_);
  parameter_pub_ = node_.advertise<std_msgs::String>("/map_debug/avoidance_parameter_edit", 10);
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

void MapDebugPanel::applyParameter(size_t index) {
  auto& row = parameters_.at(index);
  if (!row.apply->isEnabled()) return;
  bool valid = false;
  const double value = row.edit->text().toDouble(&valid);
  const bool minimum_height = row.name == "obstacle_min_height_m";
  const bool signed_height = row.name == "ground_reference_z_m";
  if (!valid || !std::isfinite(value) || (signed_height ? std::abs(value) > 1. : minimum_height ? value < 0. : value <= 0.)) {
    parameter_message_->setText(signed_height ? QString::fromUtf8("-1~1 m 범위의 숫자를 입력하세요.") : minimum_height
        ? QString::fromUtf8("0 이상의 유한한 숫자를 입력하세요.")
        : QString::fromUtf8("0보다 큰 유한한 숫자를 입력하세요."));
    return;
  }
  if (parameter_pub_.getNumSubscribers() == 0) {
    parameter_message_->setText(QString::fromUtf8("map_debug 노드 연결을 기다리는 중입니다."));
    return;
  }
  QJsonObject request;
  request.insert("name", row.name);
  request.insert("value", value);
  std_msgs::String message;
  message.data = QJsonDocument(request).toJson(QJsonDocument::Compact).toStdString();
  row.edit->setModified(true);
  parameter_pub_.publish(message);
  parameter_message_->setText(QString::fromUtf8("%1 = %2 적용 요청").arg(row.name).arg(value));
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
  const auto ground = data.value("ground_filter").toObject();
  const auto ground_mode = ground.value("mode").toString();
  if (ground.isEmpty()) {
    ground_status_->setText(QString::fromUtf8("노면 필터: 최신 점군 대기"));
  } else if (ground_mode == "imu_missing_or_stale") {
    ground_status_->setText(QString::fromUtf8("노면 필터: IMU 누락/지연 — 기존 높이 판정 유지"));
  } else if (ground_mode != "ground_filtered" && ground_mode != "ground_unconfirmed") {
    const QMap<QString, QString> reasons = {
      {"waiting_cloud", QString::fromUtf8("라이다 점군 수신 대기")},
      {"waiting_lane_geometry", QString::fromUtf8("차선 지도 수신 대기")},
      {"wrong_cloud_frame", QString::fromUtf8("라이다 frame 불일치")},
      {"cloud_time_mismatch", QString::fromUtf8("라이다 측정 시각 불일치/지연")},
      {"waiting_pose", QString::fromUtf8("라이다 시각에 맞는 GPS/IMU 위치 대기")}};
    ground_status_->setText(QString::fromUtf8("초록 점 표시 대기: ")+reasons.value(ground_mode, ground_mode));
  } else {
    QString text = QString::fromUtf8("IMU 보정: pitch %1° / roll %2° / 시간 차이 %3 ms\n")
        .arg(ground.value("pitch_deg").toDouble(), 0, 'f', 2)
        .arg(ground.value("roll_deg").toDouble(), 0, 'f', 2)
        .arg(ground.value("imu_skew_s").toDouble()*1000., 0, 'f', 1);
    text += ground_mode == "ground_filtered"
        ? QString::fromUtf8("노면 %1개 구간 / 제거 %2점 / 남긴 후보 %3점")
            .arg(ground.value("planes").toInt()).arg(ground.value("removed_points").toInt())
            .arg(ground.value("obstacle_points").toInt())
        : QString::fromUtf8("노면 미확인 — 해당 구간 기존 높이 판정 유지");
    ground_status_->setText(text);
  }
  const auto config = data.value("avoidance_parameters").toArray();
  if (config.isEmpty()) {
    for (auto& row : parameters_) row.current->setText(QString::fromUtf8("연결 대기"));
  }
  for (const auto& item : config) {
    const auto setting = item.toObject();
    for (auto& row : parameters_) {
      if (row.name != setting.value("name").toString()) continue;
      const auto value = setting.value("value");
      const auto pending = setting.value("pending_value");
      const auto description = setting.value("description").toString();
      row.label->setText(row.name + "\n" + description);
      row.label->setToolTip(description);
      QString text = value.isDouble() ? QString::number(value.toDouble(), 'g', 6)
          + " " + setting.value("unit").toString() : QString::fromUtf8("대기 중");
      text += setting.value("applied").toBool()
          ? QString::fromUtf8("\n적용됨") : QString::fromUtf8("\n주행 노드 대기");
      if (pending.isDouble()) text += QString::fromUtf8("\n→ %1 대기").arg(pending.toDouble());
      const auto error = setting.value("error").toString();
      if (!error.isEmpty()) text += "\n" + error;
      row.current->setText(text);
      row.current->setToolTip(text);
      if (value.isDouble() && setting.value("applied").toBool() && !pending.isDouble()
          && row.edit->text().toDouble() == value.toDouble()) row.edit->setModified(false);
      if (value.isDouble() && !row.edit->hasFocus() && !row.edit->isModified()
          && !pending.isDouble()) row.edit->setText(QString::number(value.toDouble(), 'g', 6));
    }
  }
  const auto parameter_error = data.value("parameter_error").toString();
  if (!parameter_error.isEmpty()) parameter_message_->setText(parameter_error);
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
  const auto width = data.value("raceline_width");
  raceline_width_->setText(width.isDouble()
      ? QString::number(width.toDouble(), 'f', 2) + " m" : waiting);
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
