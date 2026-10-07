"""Exercise real inference camera-switch methods without loading a YOLO model."""
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch


class Publisher:
    def __init__(self):self.rows=[]
    def publish(self,m):self.rows.append(m.data)


class CameraSelectionTests(unittest.TestCase):
    def setUp(self):
        string=lambda **kw:types.SimpleNamespace(**kw)
        modules={'rospy':types.SimpleNamespace(Time=types.SimpleNamespace(now=lambda:types.SimpleNamespace(to_sec=lambda:100.))),
                 'cv_bridge':types.SimpleNamespace(CvBridge=object),
                 'sensor_msgs.msg':types.SimpleNamespace(Image=object),
                 'std_msgs.msg':types.SimpleNamespace(String=string)}
        path=Path(__file__).resolve().parents[2]/'perception/scripts/traffic_light_test.py'
        spec=importlib.util.spec_from_file_location('leo_camera_switch_test',path)
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,modules):spec.loader.exec_module(module)
        self.obj=module.TrafficLightTest.__new__(module.TrafficLightTest)
        self.obj.lock=threading.RLock();self.obj.camera='front';self.obj.generation=0
        self.obj.latest=None;self.obj.auxiliary=object();self.obj.frame_number=0
        self.obj.publisher=Publisher()

    def test_switch_drops_old_buffer_and_rejects_inflight_old_result(self):
        o=self.obj;o.receive(object(),'front');self.assertIsNotNone(o.latest)
        o.select_camera(types.SimpleNamespace(data='up'))
        self.assertIsNone(o.latest)
        self.assertFalse(o.publish_result(None,[{'label':'green'}],True,0))
        self.assertEqual(len(o.publisher.rows),1)
        import json
        self.assertEqual(json.loads(o.publisher.rows[0])['camera'],'up')
        self.assertFalse(json.loads(o.publisher.rows[0])['valid'])

    def test_only_selected_camera_buffers_and_repeated_selector_does_not_reset(self):
        o=self.obj;o.select_camera(types.SimpleNamespace(data='up'))
        o.receive('wrong','front');self.assertIsNone(o.latest)
        o.receive('correct','up');self.assertEqual(o.latest[0],'correct')
        o.select_camera(types.SimpleNamespace(data='up'))
        self.assertEqual(o.generation,1);self.assertEqual(o.latest[0],'correct')

    def test_return_to_front_still_rejects_result_from_first_front_generation(self):
        o=self.obj
        for camera in ('up','front'):o.select_camera(types.SimpleNamespace(data=camera))
        self.assertFalse(o.publish_result(None,[],True,0))
        self.assertTrue(o.publish_result(None,[],True,2))
