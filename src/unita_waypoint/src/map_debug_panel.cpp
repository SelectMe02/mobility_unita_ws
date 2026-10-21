#include <unita_waypoint/map_debug_panel.h>

#include <pluginlib/class_list_macros.h>
#include <rviz/display.h>
#include <rviz/properties/property.h>
#include <rviz/render_panel.h>
#include <rviz/tool_manager.h>
#include <rviz/view_controller.h>
#include <rviz/view_manager.h>
#include <rviz/visualization_manager.h>
#include <QColor>
#include <QComboBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QHeaderView>
#include <QLabel>
#include <QLineEdit>
#include <QMap>
#include <QPixmap>
#include <QPushButton>
#include <QSplitter>
#include <QTableWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <cmath>
#include <cstdint>
#include <limits>

namespace unita_waypoint {

MapDebugPanel::MapDebugPanel(QWidget* parent)
    : rviz::Panel(parent), lidar_panel_(nullptr), lidar_manager_(nullptr) {
  setMinimumWidth(900);
  auto* root_layout = new QHBoxLayout(this);
  root_layout->setContentsMargins(2, 2, 2, 2);
  auto* splitter = new QSplitter(Qt::Horizontal, this);
  splitter->setChildrenCollapsible(false);
  root_layout->addWidget(splitter);
  auto* controls = new QWidget(splitter);
  controls->setMinimumWidth(400);
  auto* layout = new QVBoxLayout(controls);
  auto* grid = new QGridLayout();
  position_ = new QLabel(QString::fromUtf8("대기 중"));
  sector_ = new QLabel(QString::fromUtf8("대기 중"));
  raceline_ = new QLabel(QString::fromUtf8("대기 중"));
  raceline_width_ = new QLabel(QString::fromUtf8("대기 중"));
  const QString names[] = {QString::fromUtf8("차량 위치"), QString::fromUtf8("현재 Sector"),
                           QString::fromUtf8("추종 Raceline"),
                           QString::fromUtf8("Raceline 폭")};
  QLabel* values[] = {position_, sector_, raceline_, raceline_width_};
  for (int row = 0; row < 4; ++row) {
    auto* name = new QLabel(names[row]);
    QFont font = name->font();
    font.setBold(true);
    name->setFont(font);
    values[row]->setWordWrap(true);
    grid->addWidget(name, row, 0);
    grid->addWidget(values[row], row, 1);
  }
  layout->addLayout(grid);
  auto* parameter_title = new QLabel(QString::fromUtf8("실시간 장애물·회피 설정"));
  parameter_title->setStyleSheet("font-weight: bold;");
  layout->addWidget(parameter_title);
  auto* parameter_hint = new QLabel(QString::fromUtf8(
      "값 입력 후 Enter 또는 적용을 누르세요. 주행 노드가 적용한 값을 표시합니다."));
  parameter_hint->setWordWrap(true);
  layout->addWidget(parameter_hint);
  const QString keys[] = {"cluster_min_size_m", "cluster_max_size_m",
      "front_range_m", "rear_range_m", "side_range_m",
      "dynamic_speed_threshold_mps", "lane_edge_band_m", "track_confirm_time_s",
      "obstacle_horizon_m", "avoidance_margin_m", "lane_change_gap_m",
      "lane_change_lateral_accel_mps2", "lane_change_max_heading_deg", "avoidance_reaction_s", "avoidance_decel_mps2",
      "avoidance_brake_limit", "avoidance_clear_hold_s",
      "restart_accel_rate_per_s", "raceline_width",
      "sector11_min_speed_kmh", "lane_change_speed_boost_kmh",
      "lane_change_cancel_scans", "lane_change_cancel_max_progress",
      "highway_idm_time_gap_s", "highway_idm_min_gap_m", "highway_idm_accel_mps2",
      "highway_idm_comfort_decel_mps2", "highway_idm_command_horizon_s",
      "highway_mobil_politeness", "highway_mobil_threshold_mps2", "highway_mobil_safe_decel_mps2",
      "highway_r3_bias_mps2", "highway_obstacle_horizon_m", "highway_change_hold_m",
      "route_max_error_m", "blackout_route_max_error_m",
      "pass_speed_kmh", "arc_lane_change_max_kmh", "route_pass_speed_kmh",
      "ftg_no_return_speed_mps", "ftg_wall_min_clearance_speed_mps",
      "ftg_near_obstacle_speed_mps", "ftg_wall_follow_min_speed_mps"};
  const int parameter_count = sizeof(keys) / sizeof(keys[0]);
  auto* table = new QTableWidget(parameter_count, 4, controls);
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

  auto* lidar = new QWidget(splitter);
  lidar->setMinimumWidth(470);
  auto* lidar_layout = new QVBoxLayout(lidar);
  auto* lidar_header = new QHBoxLayout();
  auto* lidar_title = new QLabel(QString::fromUtf8("3D 라이다 · 인지와 판단"));
  lidar_title->setStyleSheet("font-weight: bold;");
  lidar_header->addWidget(lidar_title, 1);
  auto* reset_camera = new QPushButton(QString::fromUtf8("차량 추종 초기화"));
  connect(reset_camera, &QPushButton::clicked, this, &MapDebugPanel::resetFollowCamera);
  lidar_header->addWidget(reset_camera);
  lidar_layout->addLayout(lidar_header);
  auto* perception_view = new QComboBox(lidar);
  perception_view->addItem(QString::fromUtf8("실제 조향 인지 (자동)"), "auto");
  perception_view->addItem(QString::fromUtf8("충돌 속도 제한 인지"), "collision");
  perception_view->setToolTip(QString::fromUtf8(
      "경로 회피와 로컬 충돌 판단이 함께 동작할 때 각각의 실제 사용 점군을 확인합니다."));
  perception_view_pub_ = node_.advertise<std_msgs::String>("/map_debug/perception_view", 1, true);
  std_msgs::String initial_view;
  initial_view.data = "auto";
  perception_view_pub_.publish(initial_view);
  connect(perception_view, QOverload<int>::of(&QComboBox::currentIndexChanged),
      this, [this, perception_view](int index) {
        std_msgs::String message;
        message.data = perception_view->itemData(index).toString().toStdString();
        perception_view_pub_.publish(message);
      });
  lidar_layout->addWidget(perception_view);
  lidar_panel_ = new rviz::RenderPanel(lidar);
  lidar_panel_->setMinimumHeight(260);
  lidar_layout->addWidget(lidar_panel_, 3);
  auto* legend = new QLabel(QString::fromUtf8(
      "<span style='color:#ff4444'>● 사용 점</span> · "
      "<span style='color:#28cc48'>● 제외 점 (초록, 투명도 0.5)</span><br/>"
      "차량 크기 박스 · 차선 경계 · 로컬패스 · 인지 장애물 박스<br/>"
      "장애물 박스: 검출 크기 · 같은 프레임 점군 높이"));
  legend->setWordWrap(true);
  lidar_layout->addWidget(legend);
  vehicle_dimensions_ = new QLabel(QString::fromUtf8("차량 크기 수신 대기"));
  vehicle_dimensions_->setWordWrap(true);
  lidar_layout->addWidget(vehicle_dimensions_);
  auto* camera_hint = new QLabel(QString::fromUtf8(
      "드래그: 회전 · Shift+드래그/가운데 버튼: 이동 · 휠: 확대/축소"));
  camera_hint->setWordWrap(true);
  lidar_layout->addWidget(camera_hint);
  auto* camera_title = new QLabel(QString::fromUtf8("후방 카메라"));
  camera_title->setStyleSheet("font-weight: bold;");
  lidar_layout->addWidget(camera_title);
  rear_view_ = new QLabel(QString::fromUtf8("영상 수신 대기"));
  rear_view_->setAlignment(Qt::AlignCenter);
  rear_view_->setMinimumSize(180, 145);
  rear_view_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
  rear_view_->setStyleSheet("background: #10151b; color: #bdc7d2; border: 1px solid #485260;");
  rear_view_->setObjectName("map_debug_rear_view");
  rear_view_->setToolTip("/camera/image/rear");
  lidar_layout->addWidget(rear_view_, 2);
  auto* signal_row = new QHBoxLayout();
  signal_light_ = new QLabel(QString::fromUtf8("● 미확인"));
  signal_action_ = new QLabel(QString::fromUtf8("판단 대기"));
  signal_metrics_ = new QLabel(QString::fromUtf8("정지선 — · 제한 —"));
  signal_light_->setObjectName("map_debug_signal_light");
  signal_action_->setObjectName("map_debug_signal_action");
  signal_metrics_->setObjectName("map_debug_signal_metrics");
  signal_metrics_->setWordWrap(true);
  signal_row->addWidget(signal_light_);
  signal_row->addWidget(signal_action_);
  signal_row->addWidget(signal_metrics_, 1);
  lidar_layout->addLayout(signal_row);
  signal_reason_ = new QLabel(QString::fromUtf8("신호 판단 수신 대기"));
  signal_reason_->setObjectName("map_debug_signal_reason");
  signal_reason_->setWordWrap(true);
  signal_reason_->setTextFormat(Qt::PlainText);
  signal_reason_->setTextInteractionFlags(Qt::TextSelectableByMouse);
  signal_reason_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
  lidar_layout->addWidget(signal_reason_);
  splitter->setSizes({440, 520});
  splitter->setStretchFactor(0, 0);
  splitter->setStretchFactor(1, 1);
  initializeLidarView();
  parameter_pub_ = node_.advertise<std_msgs::String>("/map_debug/avoidance_parameter_edit", 10);
  subscriber_ = node_.subscribe("/map_debug/status", 1, &MapDebugPanel::onStatus, this);
  camera_subscribers_.push_back(node_.subscribe("/camera/image/rear", 1, &MapDebugPanel::onRearImage, this));
  camera_subscribers_.push_back(node_.subscribe("/leo/signal_status", 1, &MapDebugPanel::onGuard, this));
  camera_subscribers_.push_back(node_.subscribe("/leo/speed_limit_kmh", 1, &MapDebugPanel::onSignalCap, this));
  timer_ = new QTimer(this);
  connect(timer_, &QTimer::timeout, this, &MapDebugPanel::refresh);
  timer_->start(100);
}

MapDebugPanel::~MapDebugPanel() {
  timer_->stop();
  subscriber_.shutdown();
  for (auto& subscriber : camera_subscribers_) subscriber.shutdown();
  // The manager must release its displays and scene before the render widget dies.
  delete lidar_manager_;
}

QSize MapDebugPanel::sizeHint() const { return QSize(980, 850); }

void MapDebugPanel::initializeLidarView() {
  lidar_manager_ = new rviz::VisualizationManager(lidar_panel_);
  lidar_panel_->initialize(lidar_manager_->getSceneManager(), lidar_manager_);
  lidar_manager_->initialize();
  lidar_manager_->setFixedFrame("map");
  lidar_manager_->getToolManager()->addTool("rviz/MoveCamera");
  const auto cloud = [this](const QString& name, const QString& topic,
                            const QColor& color, double alpha, int pixels) {
    auto* display = lidar_manager_->createDisplay("rviz/PointCloud2", name, true);
    if (!display) return;
    display->subProp("Topic")->setValue(topic);
    display->subProp("Queue Size")->setValue(1);
    display->subProp("Style")->setValue("Points");
    display->subProp("Size (Pixels)")->setValue(pixels);
    display->subProp("Position Transformer")->setValue("XYZ");
    display->subProp("Color Transformer")->setValue("FlatColor");
    display->subProp("Color")->setValue(color);
    display->subProp("Alpha")->setValue(alpha);
    // Show one scan: overlapping copies would change the requested 0.5 alpha.
    display->subProp("Decay Time")->setValue(0.0);
  };
  // Smaller discarded dots reduce overlap, which otherwise makes alpha .5 look opaque.
  cloud("Discarded LiDAR points", "/map_debug/discarded_points", QColor(40, 204, 72), 0.5, 1);
  cloud("Used LiDAR points", "/map_debug/used_points", QColor(255, 0, 0), 1.0, 3);
  const auto markers = [this](const QString& name, const QString& topic) {
    auto* display = lidar_manager_->createDisplay("rviz/MarkerArray", name, true);
    if (display) display->subProp("Marker Topic")->setValue(topic);
  };
  markers("Vehicle and lane geometry", "/map_debug/lidar_scene");
  markers("Detected obstacles", "/map_debug/lidar_obstacles");
  // RViz's billboard Path display sets a zero-sized OGRE chain on empty paths.
  // Explicit LINE_STRIP/DELETE markers preserve line width and safely clear it.
  auto* path = lidar_manager_->createDisplay("rviz/Marker", "Actual local path", true);
  if (path) {
    path->subProp("Marker Topic")->setValue("/map_debug/local_path_markers");
    path->subProp("Queue Size")->setValue(1);
  }
  resetFollowCamera();
  lidar_manager_->startUpdate();
}

void MapDebugPanel::resetFollowCamera() {
  if (!lidar_manager_) return;
  auto* views = lidar_manager_->getViewManager();
  views->setCurrentViewControllerType("rviz/ThirdPersonFollower");
  auto* view = views->getCurrent();
  if (!view) return;
  view->subProp("Target Frame")->setValue("map_debug_vehicle");
  view->subProp("Distance")->setValue(32.0);
  view->subProp("Pitch")->setValue(0.65);
  view->subProp("Yaw")->setValue(3.141592654);
  view->subProp("Focal Point")->subProp("X")->setValue(6.0);
  view->subProp("Focal Point")->subProp("Y")->setValue(0.0);
  view->subProp("Focal Point")->subProp("Z")->setValue(0.0);
  lidar_manager_->queueRender();
}

void MapDebugPanel::onStatus(const std_msgs::String::ConstPtr& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  status_ = message->data;
  status_received_ = ros::WallTime::now();
}

void MapDebugPanel::onRearImage(const sensor_msgs::Image::ConstPtr& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  rear_image_ = message;
  rear_received_ = ros::WallTime::now();
}

void MapDebugPanel::onGuard(const std_msgs::String::ConstPtr& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  guard_ = message->data;
  guard_received_ = ros::WallTime::now();
}

void MapDebugPanel::onSignalCap(const std_msgs::Float64::ConstPtr& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  signal_cap_kmh_ = message->data;
  cap_received_ = ros::WallTime::now();
}

namespace {
// The bridge and detector publish 8-bit images. Own the pixels independently
// of the ROS callback and preserve row padding without adding cv_bridge.
QImage cameraFrame(const sensor_msgs::Image& message) {
  QImage::Format format;
  int channels;
  const auto& encoding = message.encoding;
  if (encoding == "rgb8" || encoding == "bgr8") {
    format = QImage::Format_RGB888;
    channels = 3;
  } else if (encoding == "rgba8" || encoding == "bgra8") {
    format = QImage::Format_RGBA8888;
    channels = 4;
  } else if (encoding == "mono8" || encoding == "8UC1") {
    format = QImage::Format_Grayscale8;
    channels = 1;
  } else {
    return QImage();
  }
  if (!message.width || !message.height ||
      message.width > static_cast<unsigned>(std::numeric_limits<int>::max() / channels) ||
      message.height > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
      message.step > static_cast<unsigned>(std::numeric_limits<int>::max()) ||
      message.step < message.width * channels ||
      static_cast<uint64_t>(message.step) * message.height > message.data.size())
    return QImage();
  QImage image(message.data.data(), message.width, message.height, message.step, format);
  return encoding == "bgr8" || encoding == "bgra8" ? image.rgbSwapped() : image.copy();
}

QString numberOrDash(const QJsonValue& value, const QString& suffix) {
  return value.isDouble() && std::isfinite(value.toDouble())
      ? QString::number(value.toDouble(), 'f', 1) + suffix : QString::fromUtf8("—");
}
}  // namespace

void MapDebugPanel::refreshCamerasAndSignal() {
  sensor_msgs::Image::ConstPtr rear;
  QJsonObject guard;
  double cap = std::numeric_limits<double>::quiet_NaN();
  const auto now = ros::WallTime::now();
  const auto fresh = [&now](const ros::WallTime& stamp, double timeout) {
    return !stamp.isZero() && (now-stamp).toSec() >= 0. && (now-stamp).toSec() <= timeout;
  };
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fresh(rear_received_, 1.5)) rear = rear_image_;
    if (fresh(guard_received_, 1.0))
      guard = QJsonDocument::fromJson(QByteArray::fromStdString(guard_)).object();
    if (fresh(cap_received_, 1.0)) cap = signal_cap_kmh_;
  }
  const auto show_image = [](QLabel* view, const sensor_msgs::Image::ConstPtr& message,
                            sensor_msgs::Image::ConstPtr& rendered, QImage& frame) {
    const double age = message ? (ros::Time::now()-message->header.stamp).toSec() : 0.;
    if (!message || message->header.stamp.isZero() || age < -.1 || age > 1.5) {
      rendered.reset();
      frame = QImage();
      view->setText(QString::fromUtf8("영상 수신 대기 / 지연"));
      return false;
    }
    if (rendered != message) {
      frame = cameraFrame(*message);
      rendered = message;
    }
    if (frame.isNull()) {
      view->setText(QString::fromUtf8("표시할 수 없는 영상 형식: %1")
          .arg(QString::fromStdString(message->encoding)));
      return false;
    }
    view->setPixmap(QPixmap::fromImage(frame).scaled(view->contentsRect().size(),
        Qt::KeepAspectRatio, Qt::SmoothTransformation));
    return true;
  };
  show_image(rear_view_, rear, rendered_rear_, rear_frame_);

  QString signal = guard.value("signal").toString();
  // Only the guard's associated head determines the lamp and STOP/GO badge.
  const QMap<QString, QString> lamp_names = {
    {"green", QString::fromUtf8("초록")}, {"red", QString::fromUtf8("빨강")},
    {"yellow", QString::fromUtf8("노랑")}};
  const QString lamp_color = signal == "green" ? "#57df8a" : signal == "red" ? "#ff6666"
      : signal == "yellow" ? "#ffd45c" : "#a4afbd";
  signal_light_->setText(QString::fromUtf8("● %1%2").arg(lamp_names.value(signal, QString::fromUtf8("미확인")))
      .arg(guard.value("signal_held").toBool() ? QString::fromUtf8(" · 유지") : QString()));
  signal_light_->setStyleSheet(QString("font-weight: bold; color: %1;").arg(lamp_color));
  const auto state = guard.value("state").toString();
  const auto decision = guard.value("decision").toString();
  QString action = QString::fromUtf8("판단 대기"), action_color = "#56616e";
  if (state == "passthrough" || state == "clear") {
    action = QString::fromUtf8("신호 제어 없음");
  } else if (decision == "stop" || state == "hold") {
    action = "STOP";
    action_color = "#9c3535";
  } else if (decision == "go" || state == "clearing" || guard.value("release_stable").toBool()) {
    action = "GO";
    action_color = "#237c49";
  }
  signal_action_->setText(action);
  signal_action_->setStyleSheet(QString("font-weight: bold; color: white; background: %1; padding: 4px 9px; border-radius: 4px;").arg(action_color));
  auto distance = guard.value("d_m");
  if (!distance.isDouble()) distance = guard.value("front_wheel_distance");
  if (!distance.isDouble()) distance = guard.value("next_controlled_stop_distance_m");
  const QString cap_text = std::isinf(cap) && cap > 0 ? QString::fromUtf8("없음")
      : std::isfinite(cap) && cap >= 0 ? QString::number(cap, 'f', 1)+" km/h" : QString::fromUtf8("—");
  signal_metrics_->setText(QString::fromUtf8("정지선 %1 · 신호 제한 %2")
      .arg(numberOrDash(distance, " m")).arg(cap_text));
  QString reason = guard.value("reason").toString();
  const QMap<QString, QString> reasons = {
    {"reviewed camera signal", QString::fromUtf8("현재 신호 판정")},
    {"entered on permitted camera signal", QString::fromUtf8("허용 신호로 진입 · 교차로 통과 중")},
    {"front already crossed on permitted signal", QString::fromUtf8("허용 신호로 정지선 통과")},
    {"no relevant signal", QString::fromUtf8("해당 신호등 미검출")},
    {"camera result missing or stale", QString::fromUtf8("카메라 인식 누락 / 지연")},
    {"stop line reached without permission", QString::fromUtf8("진행 허가 없이 정지선 도달")},
    {"checkpoint segment owned by pass-through follower", QString::fromUtf8("Raceline 주행 구간")}};
  reason = reasons.value(reason, reason);
  if (guard.value("signal_held").toBool()) reason = QString::fromUtf8("미검출 동안 직전 허용 신호 유지");
  const auto light_id = guard.value("traffic_light_id").toString();
  signal_reason_->setText(guard.isEmpty() ? QString::fromUtf8("신호 판단 수신 대기 / 지연")
      : (light_id.isEmpty() ? QString() : light_id+" · ")+reason);
  signal_reason_->setToolTip(QString::fromUtf8(QJsonDocument(guard).toJson(QJsonDocument::Indented)));
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
  refreshCamerasAndSignal();
  const auto vehicle = data.value("vehicle_dimensions").toObject();
  if (vehicle.value("front").isDouble() && vehicle.value("rear").isDouble()
      && vehicle.value("half_width").isDouble()) {
    QString dimensions = QString::fromUtf8("차량: 길이 %1 × 폭 %2 m / ")
        .arg(vehicle.value("front").toDouble()+vehicle.value("rear").toDouble(), 0, 'f', 3)
        .arg(vehicle.value("half_width").toDouble()*2., 0, 'f', 3);
    dimensions += vehicle.value("height").isDouble()
        ? QString::fromUtf8("높이 %1 m").arg(vehicle.value("height").toDouble(), 0, 'f', 3)
        : QString::fromUtf8("높이 미설정 (평면 외곽 표시)");
    vehicle_dimensions_->setText(dimensions);
  } else {
    vehicle_dimensions_->setText(QString::fromUtf8("차량 크기 수신 대기"));
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
  position_->setText(available ? QString("X %1 m, Y %2 m")
      .arg(data.value("x").toDouble(), 0, 'f', 2)
      .arg(data.value("y").toDouble(), 0, 'f', 2) : waiting);
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
}

}  // namespace unita_waypoint

PLUGINLIB_EXPORT_CLASS(unita_waypoint::MapDebugPanel, rviz::Panel)
