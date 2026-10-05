#include <unita_waypoint/map_tunner_tool.h>

#include <pluginlib/class_list_macros.h>
#include <rviz/display_context.h>
#include <rviz/geometry.h>
#include <rviz/viewport_mouse_event.h>

#include <OgrePlane.h>
#include <OgreVector3.h>
#include <QCursor>

namespace unita_waypoint {

MapTunnerTool::MapTunnerTool() {
  setName(QString::fromUtf8("경로점 클릭 편집"));
  setCursor(QCursor(Qt::CrossCursor));
}

void MapTunnerTool::onInitialize() {
  click_pub_ = node_.advertise<geometry_msgs::PointStamped>("/clicked_point", 10);
  smooth_pub_ = node_.advertise<geometry_msgs::PointStamped>("/map_tunner/smooth_point", 10);
}

void MapTunnerTool::activate() {
  setStatus(QString::fromUtf8("지도 좌클릭: 현재 모드의 추가·선택·보간·스무딩·속도 구간 선택"));
}

void MapTunnerTool::deactivate() {}

int MapTunnerTool::processMouseEvent(rviz::ViewportMouseEvent& event) {
  if (!event.leftUp() && !event.rightUp()) {
    return 0;
  }
  Ogre::Vector3 position;
  Ogre::Plane plane(Ogre::Vector3::UNIT_Z, 0.0f);
  if (!rviz::getPointOnPlaneFromWindowXY(event.viewport, plane,
                                         event.x, event.y, position)) {
    return 0;
  }
  geometry_msgs::PointStamped message;
  message.header.stamp = ros::Time::now();
  message.header.frame_id = context_->getFixedFrame().toStdString();
  message.point.x = position.x;
  message.point.y = position.y;
  message.point.z = 0.0;
  if (event.leftUp()) {
    click_pub_.publish(message);
  } else {
    smooth_pub_.publish(message);
  }
  return Render;
}

}  // namespace unita_waypoint

PLUGINLIB_EXPORT_CLASS(unita_waypoint::MapTunnerTool, rviz::Tool)
