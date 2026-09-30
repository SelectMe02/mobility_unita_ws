#!/usr/bin/env python3
"""Hold W for 30 km/h, steer with A/D, and brake on W release or S."""

import math
import signal
import sys
import time

import rospy
from morai_msgs.msg import CtrlCmd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from std_msgs.msg import String


TARGET_SPEED_KMH = 30.0
CONTROL_HZ = 20
DRIVE_KEYS = {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D}


def make_command(held_keys, max_steering_rad):
    """Use velocity control only while W is held and S is not held."""
    cmd = CtrlCmd()
    cmd.steering = max_steering_rad * (
        int(Qt.Key_A in held_keys) - int(Qt.Key_D in held_keys))
    cmd.acceleration = 0.0
    if Qt.Key_W in held_keys and Qt.Key_S not in held_keys:
        cmd.longlCmdType = 2  # MORAI velocity command, in km/h.
        cmd.velocity = TARGET_SPEED_KMH
        cmd.accel = 0.0
        cmd.brake = 0.0
    else:
        cmd.longlCmdType = 1  # Pedal command: full brake, including at startup.
        cmd.velocity = 0.0
        cmd.accel = 0.0
        cmd.brake = 1.0
    return cmd


class DriveWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.held_keys = set()
        self.max_steering_rad = float(rospy.get_param("~max_steering_rad", 0.5))
        if not math.isfinite(self.max_steering_rad) or not 0 < self.max_steering_rad <= math.pi / 2:
            raise ValueError("max_steering_rad must be between 0 and pi/2")
        self.publisher = rospy.Publisher(
            rospy.get_param("~ctrl_topic", "/ctrl_cmd"), CtrlCmd, queue_size=1)
        self.transport_status = None
        self.transport_status_time = None
        self.status_subscriber = rospy.Subscriber(
            "/control/udp_status", String, self.on_transport_status, queue_size=1)

        self.setWindowTitle("30run - WASD manual drive")
        self.setFocusPolicy(Qt.StrongFocus)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("W: 30 km/h  |  A/D: steering  |  S: brake"))
        layout.addWidget(QLabel("Keep this window focused while driving."))
        self.status = QLabel()
        layout.addWidget(self.status)
        self.bridge_status = QLabel()
        layout.addWidget(self.bridge_status)
        self.resize(540, 140)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.publish_command)
        self.timer.start(round(1000 / CONTROL_HZ))
        self.publish_command()

    def publish_command(self):
        if rospy.is_shutdown():
            self.close()
            return
        cmd = make_command(self.held_keys, self.max_steering_rad)
        self.publisher.publish(cmd)
        self.status.setText(
            "30 km/h" if cmd.longlCmdType == 2 else "BRAKE")
        if (self.transport_status_time is None
                or time.monotonic() - self.transport_status_time > 1.0):
            self.bridge_status.setText("CONTROL BRIDGE OFFLINE: no MORAI UDP control")
        else:
            self.bridge_status.setText(self.transport_status)

    def on_transport_status(self, message):
        # ROS callbacks run in another thread; the Qt timer updates the label.
        self.transport_status = message.data
        self.transport_status_time = time.monotonic()

    def brake_now(self):
        self.held_keys.clear()
        self.publisher.publish(make_command(self.held_keys, self.max_steering_rad))
        self.status.setText("BRAKE")

    def keyPressEvent(self, event):
        if not event.isAutoRepeat() and event.key() in DRIVE_KEYS:
            self.held_keys.add(event.key())
            self.publish_command()
            event.accept()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat() and event.key() in DRIVE_KEYS:
            self.held_keys.discard(event.key())
            self.publish_command()
            event.accept()
        else:
            super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        self.brake_now()
        super().focusOutEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.brake_now()
        super().closeEvent(event)


def main():
    rospy.init_node("run30", disable_signals=True)
    app = QApplication(rospy.myargv(argv=sys.argv))
    window = DriveWindow()
    window.show()
    window.activateWindow()
    window.setFocus()
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    rospy.loginfo("30run: focus the control window; hold W to drive at 30 km/h")
    try:
        return app.exec_()
    finally:
        window.brake_now()
        rospy.signal_shutdown("30run stopped")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, rospy.ROSInterruptException) as error:
        rospy.logfatal("30run failed: %s", error)
        sys.exit(1)
