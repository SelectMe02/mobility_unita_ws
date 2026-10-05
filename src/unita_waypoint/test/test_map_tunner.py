#!/usr/bin/env python3
"""Check exact sector import and independent raceline mutations."""

import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from map_tunner_core import (ProjectEditor, create_project, insert_index,
                             load_checkpoints, load_project, save_project)
from map_tunner_map import AerialMap


class MapTunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = ROOT / 'config'
        cls.original = create_project(config / 'waypoints.csv',
                                      config / 'map_tunner_checkpoints.yaml')

    def test_all_csv_waypoints_are_in_exactly_ordered_sectors(self):
        sectors = self.original['sectors']
        self.assertEqual(len(sectors), 15)
        self.assertEqual([sector['source_indices'] for sector in sectors],
                         [[0, 146], [146, 378], [378, 483], [483, 704],
                          [704, 961], [961, 1185], [1185, 1297],
                          [1297, 1745], [1745, 1895], [1895, 2275],
                          [2275, 3231], [3231, 3529], [3529, 3768],
                          [3768, 4045], [4045, 4429]])
        self.assertTrue(all(sector['lane_count'] == 1 for sector in sectors))
        self.assertEqual(sum(len(sector['racelines']['1']) - 1 for sector in sectors),
                         4429)

    def test_lane_add_remove_and_point_edits_stay_in_one_raceline(self):
        editor = ProjectEditor(copy.deepcopy(self.original))
        before_1 = copy.deepcopy(editor.lane(1, 1))
        before_2 = copy.deepcopy(editor.lane(2, 1))
        self.assertEqual(editor.add_lane(1), 2)
        self.assertEqual(editor.lane(1, 2), [])
        point = editor.add_point(1, 2, 1, 2, 3, 20)
        editor.change_point(1, 2, point['id'], speed_kmh=21, x=4)
        self.assertEqual(editor.lane(1, 2)[0]['speed_kmh'], 21)
        self.assertEqual(editor.lane(1, 2)[0]['x'], 4)
        self.assertEqual(editor.lane(1, 1), before_1)
        self.assertEqual(editor.lane(2, 1), before_2)
        self.assertEqual(editor.remove_lane(1), 2)
        self.assertEqual(editor.sector(1)['lane_count'], 1)
        self.assertTrue(editor.undo())
        self.assertEqual(editor.lane(1, 2)[0]['id'], point['id'])

    def test_round_trip_and_segment_insertion(self):
        editor = ProjectEditor(copy.deepcopy(self.original))
        points = editor.lane(1, 1)
        first, second = points[0], points[1]
        middle_x = (first['x'] + second['x']) / 2
        middle_y = (first['y'] + second['y']) / 2
        self.assertEqual(insert_index(points, middle_x, middle_y), 1)
        editor.add_point(1, 1, middle_x, middle_y, first['z'], 25, 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'project.json'
            save_project(path, editor.project)
            loaded = load_project(path)
        self.assertEqual(loaded, editor.project)

    def test_speed_range_is_inclusive_local_and_one_undo_step(self):
        editor = ProjectEditor(copy.deepcopy(self.original))
        lane = editor.lane(3, 1)
        original = copy.deepcopy(lane)
        other_sector = copy.deepcopy(editor.lane(4, 1))
        first_id, last_id = lane[3]['id'], lane[7]['id']
        self.assertEqual(editor.set_speed_range(3, 1, last_id, first_id, 37), 5)
        self.assertEqual([point['speed_kmh'] for point in lane[:3]], [20] * 3)
        self.assertEqual([point['speed_kmh'] for point in lane[3:8]], [37] * 5)
        self.assertEqual(lane[8:], original[8:])
        self.assertEqual(editor.lane(4, 1), other_sector)
        with self.assertRaises(ValueError):
            editor.set_speed_range(3, 1, first_id, other_sector[0]['id'], 40)
        with self.assertRaises(ValueError):
            editor.set_speed_range(3, 1, first_id, last_id, 201)
        self.assertTrue(editor.undo())
        self.assertEqual(editor.lane(3, 1), original)

    def test_aerial_waypoint_anchor_round_trip(self):
        config = ROOT / 'config'
        checkpoints = load_checkpoints(config / 'map_tunner_checkpoints.yaml')
        with tempfile.TemporaryDirectory() as folder:
            calibration = Path(folder) / 'calibration.yaml'
            shutil.copyfile(config / 'map_tunner_calibration.yaml', calibration)
            aerial = AerialMap(config / 'kcity_map_sim.png', calibration, checkpoints)
            point = self.original['sectors'][9]['racelines']['1'][50]
            position = (point['x'], point['y'])
            clicked = (position[0] + 2, position[1] - 3)
            expected_pixel = aerial.world_to_pixel([clicked])[0]
            actual_pixel = aerial.update_waypoint_anchor(point['id'], position, clicked)
            self.assertAlmostEqual(actual_pixel[0], expected_pixel[0], places=2)
            self.assertAlmostEqual(actual_pixel[1], expected_pixel[1], places=2)
            aerial.save_calibration()
            reloaded = AerialMap(config / 'kcity_map_sim.png', calibration, checkpoints)
            self.assertIn('point_{}'.format(point['id']), reloaded.point_anchors)

    def test_aerial_rejects_distant_checkpoint_click_and_covers_photo(self):
        config = ROOT / 'config'
        checkpoints = load_checkpoints(config / 'map_tunner_checkpoints.yaml')
        aerial = AerialMap(config / 'kcity_map_sim.png',
                           config / 'map_tunner_calibration.yaml', checkpoints)
        start_before = list(aerial.anchors['start'])
        with self.assertRaises(ValueError):
            aerial.update_checkpoint_anchor('start', checkpoints['9']['position'][:2])
        self.assertEqual(aerial.anchors['start'], start_before)
        origin_x, origin_y, raster = aerial.raster(
            [(point['x'], point['y'], point['z'])
             for sector in self.original['sectors']
             for point in sector['racelines']['1']])
        corners = np.array([[0, 0], [aerial.image.shape[1], 0],
                            [0, aerial.image.shape[0]],
                            [aerial.image.shape[1], aerial.image.shape[0]]], dtype=float)
        world = (corners - aerial.affine[2]) @ np.linalg.inv(aerial.affine[:2])
        self.assertLess(origin_x, world[:, 0].min())
        self.assertLess(origin_y, world[:, 1].min())
        self.assertGreater(origin_x + raster.shape[1] * aerial.resolution,
                           world[:, 0].max())
        self.assertGreater(origin_y + raster.shape[0] * aerial.resolution,
                           world[:, 1].max())

    def test_photo_uses_one_affine_transform_and_paint_resize_handles(self):
        config = ROOT / 'config'
        aerial = AerialMap(config / 'kcity_map_sim.png',
                           config / 'map_tunner_calibration.yaml',
                           load_checkpoints(config / 'map_tunner_checkpoints.yaml'))
        self.assertIsNone(aerial.correction)
        before = aerial.image_corners_world().copy()
        pixel_corners = np.array([[0, 0], [aerial.image.shape[1], 0],
                                  [aerial.image.shape[1], aerial.image.shape[0]],
                                  [0, aerial.image.shape[0]]])
        self.assertTrue(np.allclose(aerial.world_to_pixel(before), pixel_corners))
        aerial.resize_handle('br', before[0] + 1.2 * (before[2] - before[0]), before)
        enlarged = aerial.image_corners_world().copy()
        self.assertTrue(np.allclose(enlarged[1] - enlarged[0],
                                    1.2 * (before[1] - before[0])))
        self.assertTrue(np.allclose(enlarged[3] - enlarged[0],
                                    1.2 * (before[3] - before[0])))
        aerial.resize_handle('right', before[0] + 1.3 * (before[1] - before[0])
                             + 0.5 * (before[3] - before[0]), before)
        stretched = aerial.image_corners_world()
        self.assertTrue(np.allclose(stretched[1] - stretched[0],
                                    1.3 * (before[1] - before[0])))
        self.assertTrue(np.allclose(stretched[3] - stretched[0],
                                    before[3] - before[0]))

    def test_color_photo_mesh_follows_saved_corners(self):
        config = ROOT / 'config'
        aerial = AerialMap(config / 'kcity_map_sim.png',
                           config / 'map_tunner_calibration.yaml',
                           load_checkpoints(config / 'map_tunner_checkpoints.yaml'))
        with tempfile.TemporaryDirectory() as folder:
            first = aerial.color_mesh_resource(folder)
            tree = ElementTree.parse(Path(first[7:]))
            namespace = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
            values = tree.find('.//c:float_array[@id="positions-array"]', namespace).text
            positions = np.asarray([float(value) for value in values.split()]).reshape(4, 3)
            self.assertTrue(np.allclose(positions[:, :2], aerial.image_corners_world()))
            self.assertEqual(tree.find('.//c:image/c:init_from', namespace).text, 'photo.png')
            self.assertTrue((Path(folder) / 'photo.png').is_file())
            corners = aerial.image_corners_world()
            aerial.set_corners(corners + [1.0, 0.0])
            self.assertNotEqual(aerial.color_mesh_resource(folder), first)

    def test_interpolated_lane_has_reference_count_and_is_independent(self):
        editor = ProjectEditor(copy.deepcopy(self.original))
        original_lane = copy.deepcopy(editor.lane(1, 1))
        other_sector = copy.deepcopy(editor.lane(2, 1))
        editor.add_lane(1)
        count = editor.fill_interpolated_lane(1, 2, (1.0, 2.0), (11.0, 22.0))
        filled = editor.lane(1, 2)
        self.assertEqual(count, len(original_lane))
        self.assertEqual(len(filled) - 2, len(original_lane) - 2)
        self.assertEqual((filled[0]['x'], filled[0]['y']), (1.0, 2.0))
        self.assertEqual((filled[-1]['x'], filled[-1]['y']), (11.0, 22.0))
        self.assertEqual(editor.lane(1, 1), original_lane)
        self.assertEqual(editor.lane(2, 1), other_sector)
        with self.assertRaises(ValueError):
            editor.fill_interpolated_lane(1, 2, (0, 0), (1, 1))
        self.assertTrue(editor.undo())
        self.assertEqual(editor.lane(1, 2), [])

    def test_smoothing_changes_only_local_xy_and_is_undoable(self):
        editor = ProjectEditor(copy.deepcopy(self.original))
        points = editor.lane(5, 1)
        index = 50
        target_id = points[index]['id']
        editor.change_point(5, 1, target_id, x=points[index]['x'] + 12)
        before = copy.deepcopy(editor.lane(5, 1))
        other_sector = copy.deepcopy(editor.lane(6, 1))
        changed = editor.smooth_neighborhood(5, 1, target_id)
        after = editor.lane(5, 1)
        self.assertEqual(changed, 29)
        self.assertLess(after[index]['x'], before[index]['x'])
        self.assertEqual(after[:index - 15], before[:index - 15])
        self.assertEqual(after[index + 16:], before[index + 16:])
        self.assertEqual([(p['z'], p['speed_kmh']) for p in after],
                         [(p['z'], p['speed_kmh']) for p in before])
        self.assertEqual(editor.lane(6, 1), other_sector)
        self.assertTrue(editor.undo())
        self.assertEqual(editor.lane(5, 1), before)


if __name__ == '__main__':
    unittest.main()
