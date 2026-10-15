import math
import unittest

import numpy as np

from unita_control.lidar_lane_guard import (
    MarkingLine, choose_steering, clearance_for_steering,
    extract_marking_lines)


class LidarLaneGuardTest(unittest.TestCase):
    @staticmethod
    def road_scene(markings=()):
        xs = np.arange(3.0, 18.0, 0.35)
        ys = np.arange(-7.0, 7.01, 0.35)
        grid_x, grid_y = np.meshgrid(xs, ys)
        road = np.column_stack((grid_x.ravel(), grid_y.ravel(),
                                np.full(grid_x.size, -1.55),
                                np.full(grid_x.size, 12.0)))
        lines = []
        for lateral in markings:
            marks_x = np.arange(3.2, 17.8, 0.25)
            lines.append(np.column_stack((marks_x,
                                          np.full(len(marks_x), lateral),
                                          np.full(len(marks_x), -1.55),
                                          np.full(len(marks_x), 95.0))))
        return np.vstack([road] + lines)

    def test_recovers_road_height_markings_but_ignores_tall_returns(self):
        scene = self.road_scene(markings=(-2.2, 2.3))
        tall = scene[-60:].copy()
        tall[:, 1] = 0.1
        tall[:, 2] = 0.4
        detected = extract_marking_lines(np.vstack((scene, tall)))
        self.assertGreaterEqual(len(detected), 2)
        self.assertTrue(any(abs(line.lateral_at(5) + 2.2) < 0.3
                            for line in detected))
        self.assertTrue(any(abs(line.lateral_at(5) - 2.3) < 0.3
                            for line in detected))
        self.assertFalse(any(abs(line.lateral_at(5)) < 1.0
                             for line in detected))

    def test_crosswalk_without_forward_extent_is_not_a_lane(self):
        scene = self.road_scene()
        y = np.arange(-5, 5, 0.05)
        crosswalk = np.column_stack((np.full(len(y), 8.0), y,
                                     np.full(len(y), -1.55),
                                     np.full(len(y), 100.0)))
        self.assertEqual(extract_marking_lines(np.vstack((scene, crosswalk))), [])

    def test_lower_reflectivity_solid_line_is_detected(self):
        scene = self.road_scene(markings=(1.8,))
        scene[-59:, 3] = 50.0
        detected = extract_marking_lines(scene)
        self.assertTrue(any(abs(line.lateral_at(5) - 1.8) < 0.3
                            for line in detected))

    def test_missing_road_returns_are_invalid_not_an_unmarked_road(self):
        with self.assertRaisesRegex(ValueError, 'empty LiDAR scan'):
            extract_marking_lines(np.zeros((0, 4)))

    def test_rejects_steering_towards_a_line_and_brakes_if_trapped(self):
        line = MarkingLine(0.0, 1.7, 3.0, 13.0, 20, 10)
        adjusted, gap = choose_steering(0.12, [line])
        self.assertLess(adjusted, 0.12)
        self.assertGreaterEqual(gap, 0.0)
        self.assertGreaterEqual(clearance_for_steering([line], adjusted), 0.0)

        trapped = MarkingLine(0.0, 0.65, 2.0, 13.0, 20, 10)
        _, gap = choose_steering(0.0, [trapped])
        self.assertLess(gap, 0.0)


if __name__ == '__main__':
    unittest.main()
