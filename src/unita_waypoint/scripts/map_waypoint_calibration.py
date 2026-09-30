#!/usr/bin/env python3
"""Visual paint-like placement of a K-City photo over MORAI waypoints."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath,
                         QPen, QPolygonF, QPixmap, QTransform)
from PyQt5.QtWidgets import (QApplication, QGraphicsEllipseItem,
                             QGraphicsPathItem, QGraphicsPixmapItem,
                             QGraphicsPolygonItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsTextItem, QGraphicsView,
                             QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_tunner_core import load_checkpoints, load_csv
from map_tunner_map import AerialMap


def world_to_scene(points):
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    return np.column_stack((-values[:, 1], -values[:, 0]))


def scene_to_world(point):
    return np.asarray((-point.y(), -point.x()), dtype=float)


class CalibrationView(QGraphicsView):
    def __init__(self, aerial, waypoints, checkpoints, status_callback):
        super().__init__()
        self.aerial = aerial
        self.status_callback = status_callback
        self.scene_data = QGraphicsScene(self)
        self.setScene(self.scene_data)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(25, 28, 34))
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMinimumSize(850, 580)
        self.drag_name = None
        self.drag_start = None
        self.base_corners = None
        self.undo_stack = []
        self.dirty = False

        pixmap = QPixmap(str(aerial.image_path))
        if pixmap.isNull():
            raise ValueError('사진을 열 수 없습니다: {}'.format(aerial.image_path))
        self.photo = QGraphicsPixmapItem(pixmap)
        self.photo.setZValue(0)
        self.photo.setOpacity(0.78)
        self.scene_data.addItem(self.photo)
        self.outline = QGraphicsPolygonItem()
        self.outline.setPen(QPen(QColor(255, 214, 40), 2))
        self.outline.setBrush(QBrush(Qt.NoBrush))
        self.outline.setZValue(5)
        self.scene_data.addItem(self.outline)

        path = QPainterPath()
        projected = world_to_scene([point[:2] for point in waypoints])
        self.projected_waypoints = projected
        if len(projected):
            path.moveTo(*projected[0])
            for x, y in projected[1:]:
                path.lineTo(float(x), float(y))
        route = QGraphicsPathItem(path)
        pen = QPen(QColor(0, 255, 240), 3)
        pen.setCosmetic(True)
        route.setPen(pen)
        route.setZValue(10)
        self.scene_data.addItem(route)

        for name, checkpoint in checkpoints.items():
            x, y = world_to_scene([checkpoint['position'][:2]])[0]
            box = QGraphicsRectItem(-7, -7, 14, 14)
            box.setPos(float(x), float(y))
            box.setBrush(QBrush(QColor(0, 180, 30)))
            box.setPen(QPen(QColor(255, 255, 255), 1))
            box.setZValue(20)
            self.scene_data.addItem(box)
            label = QGraphicsTextItem(name.upper())
            label.setDefaultTextColor(QColor(255, 255, 255))
            label.setFont(QFont('Sans', 9, QFont.Bold))
            label.setPos(float(x) + 7, float(y) - 12)
            label.setZValue(21)
            self.scene_data.addItem(label)

        self.handles = {}
        for name in ('tl', 'tr', 'br', 'bl', 'top', 'right', 'bottom', 'left'):
            handle = QGraphicsRectItem(-7, -7, 14, 14)
            handle.setFlag(QGraphicsRectItem.ItemIgnoresTransformations, True)
            handle.setBrush(QBrush(QColor(255, 194, 0) if len(name) == 2
                                   else QColor(46, 153, 255)))
            handle.setPen(QPen(QColor(20, 20, 20), 1))
            handle.setZValue(30)
            self.scene_data.addItem(handle)
            self.handles[name] = handle
        self.rotate_handle = QGraphicsEllipseItem(-8, -8, 16, 16)
        self.rotate_handle.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
        self.rotate_handle.setBrush(QBrush(QColor(234, 76, 205)))
        self.rotate_handle.setPen(QPen(QColor(255, 255, 255), 1))
        self.rotate_handle.setZValue(31)
        self.scene_data.addItem(self.rotate_handle)
        self.update_photo()
        QTimer.singleShot(0, self.fit_all)

    def photo_scene_corners(self):
        return world_to_scene(self.aerial.image_corners_world())

    def handle_positions(self):
        corners = self.photo_scene_corners()
        top, right, bottom, left = ((corners[0] + corners[1]) / 2,
                                    (corners[1] + corners[2]) / 2,
                                    (corners[2] + corners[3]) / 2,
                                    (corners[3] + corners[0]) / 2)
        center = corners.mean(axis=0)
        away = top - center
        away /= max(np.linalg.norm(away), 1.0)
        return {'tl': corners[0], 'tr': corners[1], 'br': corners[2],
                'bl': corners[3], 'top': top, 'right': right,
                'bottom': bottom, 'left': left,
                'rotate': top + away * 38.0}

    def update_photo(self):
        corners = self.photo_scene_corners()
        origin = corners[0]
        u = (corners[1] - origin) / self.photo.pixmap().width()
        v = (corners[3] - origin) / self.photo.pixmap().height()
        self.photo.setTransform(QTransform(float(u[0]), float(u[1]), 0.0,
                                           float(v[0]), float(v[1]), 0.0,
                                           float(origin[0]), float(origin[1]), 1.0))
        self.outline.setPolygon(QPolygonF([QPointF(float(x), float(y))
                                           for x, y in corners]))
        for name, point in self.handle_positions().items():
            item = self.rotate_handle if name == 'rotate' else self.handles[name]
            item.setPos(float(point[0]), float(point[1]))
        all_points = np.vstack((corners, self.projected_waypoints))
        self.scene_data.setSceneRect(float(all_points[:, 0].min() - 60),
                                     float(all_points[:, 1].min() - 60),
                                     float(np.ptp(all_points[:, 0]) + 120),
                                     float(np.ptp(all_points[:, 1]) + 120))
        self.status_callback()

    def fit_all(self):
        self.fitInView(self.scene_data.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        self.scale(1.18 if event.angleDelta().y() > 0 else 1 / 1.18,
                   1.18 if event.angleDelta().y() > 0 else 1 / 1.18)

    def _pick_handle(self, viewport_point):
        for name, scene_point in self.handle_positions().items():
            on_screen = self.mapFromScene(float(scene_point[0]), float(scene_point[1]))
            if (on_screen - viewport_point).manhattanLength() <= 18:
                return name
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = event
            super().mousePressEvent(fake)
            return
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        name = self._pick_handle(event.pos())
        scene_point = self.mapToScene(event.pos())
        polygon = QPolygonF([QPointF(float(x), float(y))
                             for x, y in self.photo_scene_corners()])
        if name is None and polygon.containsPoint(scene_point, Qt.OddEvenFill):
            name = 'move'
        if name is None:
            super().mousePressEvent(event)
            return
        self.drag_name = name
        self.drag_start = scene_point
        self.base_corners = self.aerial.image_corners_world().copy()
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_name is None:
            super().mouseMoveEvent(event)
            return
        point = self.mapToScene(event.pos())
        base = self.base_corners
        try:
            if self.drag_name == 'move':
                delta = scene_to_world(point) - scene_to_world(self.drag_start)
                self.aerial.set_corners(base + delta)
            elif self.drag_name == 'rotate':
                center = base.mean(axis=0)
                first = scene_to_world(self.drag_start) - center
                second = scene_to_world(point) - center
                angle = math.atan2(second[1], second[0]) - math.atan2(first[1], first[0])
                rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]])
                self.aerial.set_corners(center + (base - center) @ rotation.T)
            else:
                self.aerial.resize_handle(self.drag_name, scene_to_world(point), base)
        except ValueError:
            return
        self.dirty = True
        self.update_photo()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self.drag_name is not None and event.button() == Qt.LeftButton:
            if not np.allclose(self.base_corners, self.aerial.image_corners_world()):
                self.undo_stack.append(self.base_corners.copy())
                self.undo_stack = self.undo_stack[-30:]
            self.drag_name = None
            self.base_corners = None
            self.unsetCursor()
            event.accept()
            return
        self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def undo(self):
        if self.undo_stack:
            self.aerial.set_corners(self.undo_stack.pop())
            self.dirty = True
            self.update_photo()


class CalibrationWindow(QMainWindow):
    def __init__(self, aerial, waypoints, checkpoints):
        super().__init__()
        self.aerial = aerial
        self.setWindowTitle('Map Waypoint Calibration · K-City')
        self.resize(1500, 850)
        central = QWidget()
        layout = QVBoxLayout(central)
        self.status = QLabel()
        instructions = QLabel(
            '사진 안쪽 드래그: 이동   |   노란 모서리: 비율 유지 확대/축소   |   '
            '파란 변 중앙: 가로/세로 늘리기   |   분홍 원: 회전   |   휠: 화면 확대')
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.view = CalibrationView(aerial, waypoints, checkpoints, self.update_status)
        layout.addWidget(self.view, 1)
        row = QHBoxLayout()
        save = QPushButton('RViz용 배치 저장 (Ctrl+S)')
        save.clicked.connect(self.save)
        undo = QPushButton('되돌리기 (Ctrl+Z)')
        undo.clicked.connect(self.view.undo)
        fit = QPushButton('전체 보기')
        fit.clicked.connect(self.view.fit_all)
        row.addWidget(save)
        row.addWidget(undo)
        row.addWidget(fit)
        row.addWidget(QLabel('사진 투명도'))
        opacity = QSlider(Qt.Horizontal)
        opacity.setRange(15, 100)
        opacity.setValue(78)
        opacity.valueChanged.connect(lambda value: self.view.photo.setOpacity(value / 100))
        row.addWidget(opacity)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.update_status()

    def update_status(self):
        if not hasattr(self, 'view'):
            return
        corners = self.aerial.image_corners_world()
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        self.status.setText('{}  |  사진 크기 {:.1f} × {:.1f} m  |  저장 파일: {}'.format(
            '저장되지 않음 ●' if self.view.dirty else '저장됨',
            width, height, self.aerial.calibration_path))

    def save(self):
        try:
            self.aerial.save_calibration()
        except OSError as error:
            QMessageBox.critical(self, '저장 실패', str(error))
            return False
        self.view.dirty = False
        self.update_status()
        return True

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_S:
            self.save()
        elif event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.view.undo()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.view.dirty:
            choice = QMessageBox.question(self, '저장되지 않은 배치',
                                          '사진 배치를 저장하고 종료할까요?',
                                          QMessageBox.Save | QMessageBox.Discard |
                                          QMessageBox.Cancel, QMessageBox.Save)
            if choice == QMessageBox.Cancel:
                event.ignore()
                return
            if choice == QMessageBox.Save:
                if not self.save():
                    event.ignore()
                    return
        event.accept()


def main():
    config = Path(__file__).resolve().parents[1] / 'config'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default=str(config / 'kcity_map_sim.png'))
    parser.add_argument('--waypoints', default=str(config / 'waypoints.csv'))
    parser.add_argument('--checkpoints', default=str(config / 'map_tunner_checkpoints.yaml'))
    parser.add_argument('--calibration', default=str(config / 'map_tunner_calibration.yaml'))
    arguments = parser.parse_args()
    app = QApplication(sys.argv[:1])
    checkpoints = load_checkpoints(arguments.checkpoints)
    aerial = AerialMap(arguments.image, arguments.calibration, checkpoints)
    window = CalibrationWindow(aerial, load_csv(arguments.waypoints), checkpoints)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
