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
  const QString names[] = {QString::fromUtf8("현재 속도"), QString::fromUtf8("차량 위치"),
                           QString::fromUtf8("헤딩"), QString::fromUtf8("현재 Sector")};
  QLabel* values[] = {speed_, position_, heading_, sector_};
  for (int row = 0; row < 4; ++row) {
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
}

void MapDebugPanel::refresh() {
  std::string status;
  {
    std::lock_guard<std::mutex> lock(mutex_);
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
}

}  // namespace unita_waypoint

PLUGINLIB_EXPORT_CLASS(unita_waypoint::MapDebugPanel, rviz::Panel)
