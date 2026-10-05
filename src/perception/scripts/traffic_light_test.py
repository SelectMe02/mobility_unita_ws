#!/usr/bin/env python3
"""Log traffic-light detections from best1.pt and a ROS camera Image topic."""

from pathlib import Path
import threading
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


SIGNALS = {
    "traffic_light_green": "초록",
    "traffic_light_green_left": "초록+좌회전",
    "traffic_light_left": "좌회전",
    "traffic_light_red": "빨강",
    "traffic_light_red_and_yellow": "빨강+노랑",
    "traffic_light_yellow": "노랑",
}


class TrafficLightTest:
    def __init__(self):
        model_path = Path(rospy.get_param("~model_path", "best1.pt")).expanduser()
        if not model_path.is_file():
            raise ValueError("모델 파일이 없습니다: %s (~model_path를 지정하세요)" % model_path)
        self.confidence = float(rospy.get_param("~confidence", 0.5))
        self.rate = float(rospy.get_param("~inference_rate", 2.0))
        self.timeout = float(rospy.get_param("~image_timeout", 2.0))
        self.device = str(rospy.get_param("~device", "cpu"))
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence는 0 초과 1 이하여야 합니다")
        if not 0 < self.rate < float("inf") or not 0 < self.timeout < float("inf"):
            raise ValueError("inference_rate와 image_timeout은 유한한 양수여야 합니다")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("실행 Python 환경에 ultralytics를 설치하세요: "
                               "python3 -m pip install ultralytics==8.4.55") from exc
        self.model = YOLO(str(model_path))
        if self.model.task != "detect":
            raise ValueError("객체 감지(detect) 모델이 필요합니다")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest = None
        self.last_received = time.monotonic()
        topic = rospy.get_param("~image_topic", "/camera/image/front")
        self.subscriber = rospy.Subscriber(
            topic, Image, self.receive, queue_size=1, buff_size=2**24)
        rospy.loginfo("신호등 테스트 시작: model=%s topic=%s classes=%s",
                      model_path, topic, self.model.names)

    def receive(self, message):
        with self.lock:
            self.last_received = time.monotonic()
            self.latest = (message, self.last_received)

    def run(self):
        # Only consume the newest image; inference does not block the subscriber.
        while not rospy.is_shutdown():
            started = time.monotonic()
            with self.lock:
                item = self.latest
                self.latest = None
                last_received = self.last_received
            if started - last_received > self.timeout:
                rospy.logwarn_throttle(5.0, "카메라 영상 수신 없음: 신호 판단 불가")
            elif item is not None:
                try:
                    self.infer(item[0], item[1])
                except Exception as exc:
                    rospy.logerr_throttle(5.0, "신호등 추론 실패: %s" % exc)
            time.sleep(max(0.0, 1.0 / self.rate - (time.monotonic() - started)))

    def infer(self, message, received_at):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        result = self.model.predict(source=frame, conf=self.confidence,
                                    device=self.device, verbose=False)[0]
        if time.monotonic() - received_at > self.timeout:
            rospy.logwarn_throttle(5.0, "영상 처리 지연: 오래된 신호 판단 결과 생략")
            return
        detections = []
        if result.boxes is not None:
            for box in result.boxes.cpu():
                label = result.names[int(box.cls.item())]
                signal = SIGNALS.get(label, "알 수 없는 신호")
                coordinates = tuple(round(value) for value in box.xyxy[0].tolist())
                detections.append("%s (%s) confidence=%.3f bbox=%s" % (
                    signal, label, float(box.conf.item()), coordinates))
        if detections:
            rospy.loginfo("신호등 %d개 | %s", len(detections), " | ".join(detections))
        else:
            rospy.loginfo("신호등 미검출 (confidence >= %.2f)", self.confidence)


def main():
    rospy.init_node("traffic_light_test")
    try:
        TrafficLightTest().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("신호등 테스트 종료: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
