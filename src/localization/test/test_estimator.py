import math
import unittest
from unita_localization.estimator import GPSIMUEstimator


class EstimatorTests(unittest.TestCase):
    def ready(self, **options):
        estimator = GPSIMUEstimator(**options)
        # UTM zone 52 central meridian, equator: easting 500000, northing 0.
        estimator.update_gps(0, 129, 499990, -20, 1, 10, 100)
        estimator.update_imu((0, 0, 2 ** 0.5, 2 ** 0.5), True, 10, 100)
        return estimator

    def test_projection_offsets_and_normalized_yaw(self):
        state, reason = self.ready().snapshot(10.1, 100.1)
        self.assertEqual(reason, '')
        self.assertAlmostEqual(state[0], 10, places=5)
        self.assertAlmostEqual(state[1], 20)
        self.assertAlmostEqual(state[2], math.pi / 2)
        self.assertEqual(state[3], 10)

    def test_wait_for_both_sensors(self):
        e = GPSIMUEstimator()
        e.update_gps(0, 129, 0, 0, 1, 10, 100)
        self.assertIsNone(e.snapshot(10, 100)[0])

    def test_timeout_ros_wall_and_future_time(self):
        e = self.ready()
        for now, wall in ((11.1, 100.1), (10.1, 101.1), (9, 100.1)):
            self.assertIsNone(e.snapshot(now, wall)[0])

    def test_timestamp_skew(self):
        e = self.ready()
        e.update_imu((0, 0, 0, 1), True, 10.5, 100.5)
        self.assertIsNone(e.snapshot(10.5, 100.5)[0])

    def test_gps_invalidates_previous_fix_and_recovers(self):
        e = self.ready()
        for lat, lon, status in ((0, 0, 1), (37, 127, 0), (float('nan'), 127, 1), (91, 127, 1)):
            with self.assertRaises(ValueError):
                e.update_gps(lat, lon, 0, 0, status, 10, 100)
            self.assertIsNone(e.snapshot(10, 100)[0])
        e.update_gps(0, 129, 0, 0, 1, 10, 100)
        self.assertIsNotNone(e.snapshot(10, 100)[0])

    def test_imu_invalidates_previous_orientation(self):
        for q, available in (((0, 0, 0, 0), True), ((0, 0, 0, 1), False),
                             ((0, 0, float('nan'), 1), True)):
            e = self.ready()
            with self.assertRaises(ValueError):
                e.update_imu(q, available, 10, 100)
            self.assertIsNone(e.snapshot(10, 100)[0])

    def test_yaw_offset(self):
        e = self.ready(yaw_offset=math.pi)
        self.assertAlmostEqual(e.snapshot(10, 100)[0][2], -math.pi / 2)

    def test_projection_failure_invalidates_previous_fix(self):
        e = self.ready()
        with self.assertRaises(ValueError):
            e.update_gps(0, 39, 0, 0, 1, 10, 100)
        self.assertIsNone(e.snapshot(10, 100)[0])

    def test_configuration_validation(self):
        for options in ({'utm_zone': 0}, {'timeout': 0}, {'max_skew': -1},
                        {'yaw_offset': float('nan')}):
            with self.assertRaises(ValueError):
                GPSIMUEstimator(**options)


if __name__ == '__main__':
    unittest.main()
