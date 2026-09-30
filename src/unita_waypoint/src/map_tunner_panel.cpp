#include <unita_waypoint/map_tunner_panel.h>

#include <pluginlib/class_list_macros.h>
#include <rviz/properties/property.h>
#include <rviz/tool.h>
#include <rviz/tool_manager.h>
#include <rviz/view_controller.h>
#include <rviz/view_manager.h>
#include <rviz/visualization_manager.h>

#include <QComboBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMetaObject>
#include <QPushButton>
#include <QSignalBlocker>
#include <QTimer>
#include <QVBoxLayout>

namespace unita_waypoint {

namespace {
QLabel* note(const QString& text) {
  auto* label = new QLabel(text);
  label->setWordWrap(true);
  return label;
}

QPushButton* button(const QString& text, const QString& tooltip) {
  auto* result = new QPushButton(text);
  result->setToolTip(tooltip);
  result->setMinimumHeight(32);
  return result;
}
}  // namespace

MapTunnerPanel::MapTunnerPanel(QWidget* parent) : rviz::Panel(parent) {
  auto* outer = new QVBoxLayout(this);
  outer->setSpacing(8);

  auto* title = note(QString::fromUtf8("K-City 경로 편집"));
  QFont title_font = title->font();
  title_font.setBold(true);
  title_font.setPointSize(title_font.pointSize() + 2);
  title->setFont(title_font);
  outer->addWidget(title);
  auto* legend = note(QString::fromUtf8(
      "초록 번호 박스 = 체크포인트   |   하늘색 선 = 전체 경로\n"
      "빨간 선 = 선택한 구간의 Raceline 1\n"
      "파란/초록 선 = 추가 Raceline   |   노란 점 = 선택한 경로점"));
  legend->setStyleSheet("QLabel { color: #294b68; font-weight: bold; }");
  outer->addWidget(legend);

  auto* sector_group = new QGroupBox(QString::fromUtf8("1. 구간과 차선"));
  auto* sector_layout = new QGridLayout(sector_group);
  sector_box_ = new QComboBox();
  for (int i = 1; i <= 15; ++i) {
    const QString from = i == 1 ? "START" : QString::number(i - 1);
    const QString to = i == 15 ? "START" : QString::number(i);
    sector_box_->addItem(QString("S%1  %2 → %3").arg(i).arg(from).arg(to));
  }
  sector_label_ = note(QString::fromUtf8("S1  START → 1"));
  lane_label_ = note(QString::fromUtf8("차선 1개"));
  add_lane_button_ = button(QString::fromUtf8("＋ 차선 추가"),
                            QString::fromUtf8("현재 구간에 빈 raceline을 추가합니다."));
  remove_lane_button_ = button(QString::fromUtf8("－ 마지막 차선 삭제"),
                               QString::fromUtf8("마지막 raceline과 점을 삭제합니다. 실행 취소 가능."));
  lane_box_ = new QComboBox();
  lane_box_->addItem("Raceline 1");
  sector_layout->addWidget(sector_box_, 0, 0, 1, 2);
  sector_layout->addWidget(sector_label_, 1, 0, 1, 2);
  sector_layout->addWidget(lane_label_, 2, 0, 1, 2);
  sector_layout->addWidget(add_lane_button_, 3, 0);
  sector_layout->addWidget(remove_lane_button_, 3, 1);
  sector_layout->addWidget(note(QString::fromUtf8("편집할 Raceline:")), 4, 0);
  sector_layout->addWidget(lane_box_, 4, 1);
  outer->addWidget(sector_group);

  auto* point_group = new QGroupBox(QString::fromUtf8("2. 경로점 편집"));
  auto* point_layout = new QGridLayout(point_group);
  add_point_button_ = button(QString::fromUtf8("＋ 점 추가"),
                             QString::fromUtf8("지도에서 좌클릭하면 현재 Raceline에 점이 추가됩니다."));
  select_point_button_ = button(QString::fromUtf8("점 선택"),
                                QString::fromUtf8("지도에서 점 근처를 좌클릭해 선택합니다."));
  interpolate_button_ = button(QString::fromUtf8("↔ 선 보간"),
                               QString::fromUtf8("빈 Raceline에서 시작점과 끝점을 좌클릭합니다."));
  smooth_button_ = button(QString::fromUtf8("〰 스무딩"),
                          QString::fromUtf8("경로점에서 우클릭하면 앞뒤 15개 점을 참조해 스무딩합니다."));
  point_label_ = note(QString::fromUtf8("선택된 점 없음"));
  speed_edit_ = new QLineEdit();
  speed_edit_->setValidator(new QIntValidator(0, 200, speed_edit_));
  speed_edit_->setPlaceholderText(QString::fromUtf8("0~200"));
  speed_apply_button_ = button(QString::fromUtf8("속도 적용 (km/h)"),
                               QString::fromUtf8("선택한 점의 속도를 입력한 정수로 저장합니다."));
  delete_point_button_ = button(QString::fromUtf8("선택한 점 삭제"),
                                QString::fromUtf8("선택한 점을 삭제합니다. 실행 취소 가능."));
  point_layout->addWidget(add_point_button_, 0, 0);
  point_layout->addWidget(select_point_button_, 0, 1);
  point_layout->addWidget(interpolate_button_, 1, 0);
  point_layout->addWidget(smooth_button_, 1, 1);
  point_layout->addWidget(note(QString::fromUtf8(
      "점 추가/선택/보간/스무딩: 지도 좌클릭\n"
      "노란 선택 점 이동: RViz 상단 Interact 도구로 끌기")), 2, 0, 1, 2);
  point_layout->addWidget(point_label_, 3, 0, 1, 2);
  point_layout->addWidget(speed_edit_, 4, 0);
  point_layout->addWidget(speed_apply_button_, 4, 1);
  point_layout->addWidget(delete_point_button_, 5, 0, 1, 2);
  outer->addWidget(point_group);

  auto* image_group = new QGroupBox(QString::fromUtf8("3. 사진 크기와 위치"));
  auto* image_layout = new QVBoxLayout(image_group);
  resize_button_ = button(QString::fromUtf8("사진 크기 조절"),
                          QString::fromUtf8("사진의 모서리 또는 변 중앙 손잡이를 드래그합니다."));
  image_layout->addWidget(resize_button_);
  image_layout->addWidget(note(QString::fromUtf8(
      "노란 모서리: 비율 유지 확대·축소\n"
      "파란 변 중앙: 가로 또는 세로만 늘리기\n"
      "손잡이를 끌어 놓으면 사진 위치가 자동 저장됩니다.")));
  outer->addWidget(image_group);

  auto* final_row = new QHBoxLayout();
  auto* undo_button = button(QString::fromUtf8("실행 취소"), QString::fromUtf8("마지막 편집을 되돌립니다."));
  save_button_ = button(QString::fromUtf8("JSON 저장"), QString::fromUtf8("편집 내용을 파일에 저장합니다."));
  final_row->addWidget(undo_button);
  final_row->addWidget(save_button_);
  outer->addLayout(final_row);

  status_label_ = note(QString::fromUtf8("map_tunner 노드를 기다리는 중…"));
  status_label_->setStyleSheet("QLabel { color: #315d89; }");
  outer->addWidget(status_label_);
  outer->addStretch();

  command_pub_ = node_.advertise<std_msgs::String>("/map_tunner/command", 10);
  state_sub_ = node_.subscribe("/map_tunner/state", 10, &MapTunnerPanel::stateCallback, this);

  connect(sector_box_, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
          this, [this](int index) { command("sector:" + std::to_string(index + 1)); });
  connect(lane_box_, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
          this, [this](int index) { command("lane:" + std::to_string(index + 1)); });
  connect(add_lane_button_, &QPushButton::clicked, this, [this]() {
    command("add_lane"); activateEditTool();
  });
  connect(remove_lane_button_, &QPushButton::clicked, this, [this]() { command("remove_lane"); });
  connect(add_point_button_, &QPushButton::clicked, this, [this]() {
    command("mode_add"); activateEditTool();
  });
  connect(select_point_button_, &QPushButton::clicked, this, [this]() {
    command("mode_select"); activateEditTool();
  });
  connect(interpolate_button_, &QPushButton::clicked, this, [this]() {
    command("mode_interpolate"); activateEditTool();
  });
  connect(smooth_button_, &QPushButton::clicked, this, [this]() {
    command("mode_smooth"); activateEditTool();
  });
  connect(resize_button_, &QPushButton::clicked, this, [this]() {
    command("mode_resize"); activateInteractTool();
  });
  connect(speed_apply_button_, &QPushButton::clicked, this, [this]() { applySpeed(); });
  connect(speed_edit_, &QLineEdit::returnPressed, this, [this]() { applySpeed(); });
  connect(delete_point_button_, &QPushButton::clicked, this, [this]() { command("delete_point"); });
  connect(undo_button, &QPushButton::clicked, this, [this]() { command("undo"); });
  connect(save_button_, &QPushButton::clicked, this, [this]() { command("save"); });
}

void MapTunnerPanel::onInitialize() {
  QTimer::singleShot(500, this, [this]() { activateEditTool(); });
}

void MapTunnerPanel::activateEditTool() {
  if (!vis_manager_ || !vis_manager_->getToolManager()) {
    return;
  }
  auto* manager = vis_manager_->getToolManager();
  for (int index = 0; index < manager->numTools(); ++index) {
    rviz::Tool* tool = manager->getTool(index);
    if (tool->getClassId() == "unita_waypoint/MapTunnerTool") {
      manager->setCurrentTool(tool);
      return;
    }
  }
  manager->setCurrentTool(manager->addTool("unita_waypoint/MapTunnerTool"));
}

void MapTunnerPanel::activateInteractTool() {
  if (!vis_manager_ || !vis_manager_->getToolManager()) {
    return;
  }
  auto* manager = vis_manager_->getToolManager();
  for (int index = 0; index < manager->numTools(); ++index) {
    rviz::Tool* tool = manager->getTool(index);
    if (tool->getClassId() == "rviz/Interact") {
      manager->setCurrentTool(tool);
      return;
    }
  }
}

void MapTunnerPanel::applySpeed() {
  if (speed_edit_->hasAcceptableInput()) {
    command("speed:" + speed_edit_->text().toStdString());
  }
}

void MapTunnerPanel::command(const std::string& value) {
  std_msgs::String message;
  message.data = value;
  command_pub_.publish(message);
}

void MapTunnerPanel::stateCallback(const std_msgs::String::ConstPtr& message) {
  const QString payload = QString::fromStdString(message->data);
  QMetaObject::invokeMethod(this, "updateState", Qt::QueuedConnection,
                            Q_ARG(QString, payload));
}

void MapTunnerPanel::updateState(const QString& json) {
  QJsonParseError error;
  const auto parsed = QJsonDocument::fromJson(json.toUtf8(), &error);
  if (error.error != QJsonParseError::NoError || !parsed.isObject()) {
    status_label_->setText(QString::fromUtf8("편집기 상태를 읽지 못했습니다."));
    return;
  }
  const auto state = parsed.object();
  const int sector = state.value("sector").toInt(1);
  const int lane_count = state.value("lane_count").toInt(1);
  const int lane = state.value("lane").toInt(1);
  const bool has_point = !state.value("point_id").isNull();
  {
    QSignalBlocker block(sector_box_);
    sector_box_->setCurrentIndex(sector - 1);
  }
  {
    QSignalBlocker block(lane_box_);
    if (lane_box_->count() != lane_count) {
      lane_box_->clear();
      for (int i = 1; i <= lane_count; ++i) {
        lane_box_->addItem(QString("Raceline %1").arg(i));
      }
    }
    lane_box_->setCurrentIndex(lane - 1);
  }
  sector_label_->setText(QString("S%1   %2 → %3").arg(sector)
      .arg(state.value("from").toString()).arg(state.value("to").toString()));
  lane_label_->setText(QString::fromUtf8("차선 %1개 · 현재 Raceline %2").arg(lane_count).arg(lane));
  add_lane_button_->setEnabled(lane_count < 3);
  remove_lane_button_->setEnabled(lane_count > 1);
  speed_edit_->setEnabled(has_point);
  speed_apply_button_->setEnabled(has_point);
  delete_point_button_->setEnabled(has_point);
  if (has_point) {
    point_label_->setText(QString::fromUtf8("선택한 점 #%1 · %2 km/h")
                          .arg(state.value("point_id").toInt())
                          .arg(state.value("speed_kmh").toInt()));
    if (!speed_edit_->hasFocus()) {
      speed_edit_->setText(QString::number(state.value("speed_kmh").toInt()));
    }
  } else {
    point_label_->setText(QString::fromUtf8("선택된 점 없음"));
    speed_edit_->clear();
  }
  const QString mode = state.value("mode").toString();
  const QString active_style = "QPushButton { background: #efad54; font-weight: bold; }";
  add_point_button_->setStyleSheet(mode == "add" ? active_style : "");
  select_point_button_->setStyleSheet(mode == "select" ? active_style : "");
  interpolate_button_->setStyleSheet(mode == "interpolate" ? active_style : "");
  smooth_button_->setStyleSheet(mode == "smooth" ? active_style : "");
  resize_button_->setStyleSheet(mode == "resize" ? active_style : "");
  interpolate_button_->setEnabled(lane > 1);
  const bool dirty = state.value("dirty").toBool();
  save_button_->setText(dirty ? QString::fromUtf8("JSON 저장 ●") : QString::fromUtf8("JSON 저장"));
  QString status = state.value("message").toString();
  if (status.isEmpty()) {
    status = dirty ? QString::fromUtf8("저장되지 않은 변경 사항이 있습니다.")
                   : QString::fromUtf8("저장된 상태입니다.");
  }
  status_label_->setText(status);
}

}  // namespace unita_waypoint

PLUGINLIB_EXPORT_CLASS(unita_waypoint::MapTunnerPanel, rviz::Panel)
