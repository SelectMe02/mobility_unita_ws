import math
import unittest
from unita_visualization.path_metrics import nearest_segment


class PathMetricsTests(unittest.TestCase):
    def test_signed_error_and_projection(self):
        p = nearest_segment([(0, 0), (10, 0)], 4, 2, .1)
        self.assertEqual(p[:3], (4, 0, 2))
        self.assertAlmostEqual(p[3], .1)
        self.assertEqual(p[4:], (4, 0))
        self.assertEqual(nearest_segment([(0, 0), (10, 0)], 4, -2, 0)[2], -2)

    def test_direction_and_heading_wrap(self):
        p = nearest_segment([(10, 0), (0, 0)], 4, 2, -math.pi + .1)
        self.assertEqual(p[2], -2)
        self.assertAlmostEqual(p[3], .1)

    def test_endpoints_and_zero_segments(self):
        p = nearest_segment([(0, 0), (0, 0), (10, 0)], 12, 0, 0)
        self.assertEqual(p[:3], (10, 0, 2))
        self.assertEqual(p[4:], (10, 1))
        with self.assertRaises(ValueError):
            nearest_segment([(0, 0), (0, 0)], 0, 0, 0)

    def test_corner_progress(self):
        p = nearest_segment([(0, 0), (10, 0), (10, 10)], 12, 6, math.pi/2)
        self.assertEqual(p[:3], (10, 6, -2))
        self.assertAlmostEqual(p[3], 0)
        self.assertEqual(p[4:], (16, 1))
