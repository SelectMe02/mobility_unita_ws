import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

path=Path(__file__).resolve().parents[1]/'scripts/master_reconnecting_process.py'
spec=importlib.util.spec_from_file_location('debug_reconnect',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ReconnectTest(unittest.TestCase):
    def test_master_restart_restores_only_debug_private_parameters(self):
        master=Mock();master.getPid.return_value=(1,'',100)
        master.getParam.side_effect=lambda caller,key: (1,'','run1' if key=='/run_id' else {'project_file':'test.json'})
        watcher=module.MasterRegistration('/map_debug')
        self.assertFalse(watcher.observe(master))
        self.assertFalse(watcher.observe(master))
        master.setParam.assert_not_called()
        master.getPid.return_value=(1,'',101)
        master.setParam.return_value=(1,'',0)
        self.assertTrue(watcher.observe(master))
        master.setParam.assert_called_once_with('/map_debug','/map_debug',{'project_file':'test.json'})
        self.assertFalse(watcher.observe(master))

    def test_unavailable_master_does_not_forget_identity(self):
        master=Mock();master.getPid.return_value=(1,'',100);master.getParam.return_value=(1,'','run1')
        watcher=module.MasterRegistration('/map_debug_rviz');watcher.observe(master)
        master.getPid.side_effect=OSError('offline')
        with self.assertRaises(OSError):watcher.observe(master)
        self.assertEqual(watcher.identity,(100,'run1'))

    def test_launch_name_and_namespace_preserved(self):
        self.assertEqual(module.node_name(['__name:=map_debug','__log:=x'],{}),'/map_debug')
        self.assertEqual(module.node_name(['__name:=map_debug','__ns:=/debug'],{}),'/debug/map_debug')
