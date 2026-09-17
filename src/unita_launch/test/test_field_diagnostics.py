import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace as Object
import unittest

from unita_visualization.field_diagnostics import (
    Check, SensorStats, aggregate, ip_check, measured_hz, sensor_detail, stamp_check)


def header(stamp=100):
    return Object(frame_id='sensor', stamp=Object(to_sec=lambda: stamp))


def gps(**values):
    fields = dict(latitude=37.24, longitude=126.77, status=1,
                  eastOffset=0.0, northOffset=0.0, header=header())
    fields.update(values)
    return Object(**fields)


def imu(quaternion=(0, 0, 0, 1), covariance=0):
    return Object(orientation=Object(**dict(zip(('x', 'y', 'z', 'w'), quaternion))),
                  orientation_covariance=[covariance] + [0] * 8, header=header())


class FieldDiagnosticTests(unittest.TestCase):
    def test_rate_uses_message_intervals(self):
        self.assertAlmostEqual(measured_hz([10, 10.02, 10.04]), 50)
        self.assertAlmostEqual(measured_hz([10, 10.1, 10.4]), 5)
        for times in ([], [10], [10, 10]):
            self.assertIsNone(measured_hz(times))

    def test_stale_future_zero_and_invalid_timestamps(self):
        self.assertEqual(stamp_check(100, 100.2, 1)[0], 'PASS')
        self.assertEqual(stamp_check(100, 102, 1)[0], 'FAIL')
        self.assertEqual(stamp_check(102, 100, 1)[0], 'FAIL')
        self.assertEqual(stamp_check(0, 100, 1)[0], 'WARN')
        for stamp in (math.nan, math.inf, -1):
            self.assertEqual(stamp_check(stamp, 100, 1)[0], 'FAIL')

    def test_gps_transport_validity_does_not_require_offsets(self):
        self.assertIn('status=1', sensor_detail('gps', gps()))
        for message in (gps(latitude=math.nan), gps(longitude=math.inf),
                        gps(latitude=0, longitude=0), gps(status=0),
                        gps(latitude=90), gps(longitude=200)):
            with self.assertRaises(ValueError):
                sensor_detail('gps', message)

    def test_imu_valid_and_invalid_quaternions(self):
        self.assertIn('norm=1.000000', sensor_detail('imu', imu()))
        self.assertIn('norm=2.000000', sensor_detail('imu', imu((0, 0, 0, 2))))
        for message in (imu((0, 0, 0, 0)), imu((0, 0, math.nan, 1)),
                        imu((0, 0, 0, math.inf)), imu(covariance=-1)):
            with self.assertRaises(ValueError):
                sensor_detail('imu', message)

    def test_pass_warn_fail_aggregation(self):
        self.assertEqual(aggregate([Check('x', 'PASS')]), ('READY', 0))
        self.assertEqual(aggregate([Check('x', 'WARN')]), ('READY WITH WARNINGS', 0))
        self.assertEqual(aggregate([Check('x', 'WARN'), Check('y', 'FAIL')]), ('NOT READY', 1))

    def test_loopback_and_invalid_addresses(self):
        self.assertEqual(ip_check('127.0.0.1').status, 'WARN')
        self.assertEqual(ip_check('127.0.0.1', False).status, 'PASS')
        self.assertEqual(ip_check('192.168.0.20').status, 'PASS')
        for address in ('SIM_PC_IP', '', '0.0.0.0', '224.0.0.1', '255.255.255.255', '::1'):
            self.assertEqual(ip_check(address).status, 'FAIL')

    def test_missing_optional_sensor_warns_without_failing(self):
        rear = SensorStats('Rear', 'camera', required=False)
        check = rear.result(100, 100, 1)
        self.assertEqual(check.status, 'WARN')
        self.assertEqual(aggregate([check])[1], 0)
        self.assertEqual(SensorStats('Front', 'camera').result(100, 100, 1).status, 'FAIL')

    def test_valid_reception_reports_rate_and_frame(self):
        stats = SensorStats('GPS', 'gps')
        stats.observe(gps(header=header(100)), 10)
        stats.observe(gps(header=header(100.1)), 10.1)
        result = stats.result(10.2, 100.2, 1)
        self.assertEqual(result.status, 'PASS')
        self.assertIn('10.0 Hz', result.detail)
        self.assertIn('frame_id=sensor', result.detail)
        self.assertIn('age=0.100s', result.detail)

    def test_frozen_timestamp_and_stopped_reception_fail(self):
        stats = SensorStats('GPS', 'gps')
        stats.observe(gps(), 10)
        stats.observe(gps(), 10.1)
        self.assertIn('not advancing', stats.result(10.2, 100.2, 1).detail)
        self.assertEqual(stats.result(12, 100.2, 1).status, 'FAIL')

    def test_invalid_optional_data_is_warn_required_is_fail(self):
        for required, level in ((True, 'FAIL'), (False, 'WARN')):
            stats = SensorStats('GPS', 'gps', required)
            stats.observe(gps(status=0), 10)
            self.assertEqual(stats.result(10.1, 100.1, 1).status, level)

    def test_image_and_cloud_payload_validation(self):
        image = Object(width=2, height=1, step=6, data=b'123456', encoding='bgr8')
        self.assertIn('2x1', sensor_detail('camera', image))
        cloud = Object(width=1, height=1, point_step=12, data=b'0' * 12)
        self.assertIn('1 points', sensor_detail('velodyne', cloud))
        for kind, message in (('camera', Object(width=0, height=1, data=b'')),
                              ('camera', Object(width=2, height=1, step=6, data=b'1')),
                              ('velodyne', Object(width=1, height=1, point_step=12, data=b'1'))):
            with self.assertRaises(ValueError):
                sensor_detail(kind, message)

    def test_ros_service_timeout_is_bounded(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/field_check.py'
        spec = importlib.util.spec_from_file_location('field_check', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import threading
        release = threading.Event()
        try:
            with self.assertRaises(TimeoutError):
                module.bounded_call(release.wait, .01)
        finally:
            release.set()
        self.assertEqual(module.bounded_call(lambda: 3, 1), 3)


if __name__ == '__main__':
    unittest.main()
