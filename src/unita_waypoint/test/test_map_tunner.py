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
                             load_checkpoints, load_project, primary_route, save_project,
                             generate_racelines, raceline_layout_settings)
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

    def test_saved_primary_route_is_ordered_without_changing_original_csv(self):
        config = ROOT / 'config'
        project = load_project(config / 'map_tunner_project.json')
        route, sectors = primary_route(project)
        self.assertGreater(len(route), 4000)
        self.assertEqual(sorted(set(sectors)), list(range(1, 16)))
        self.assertEqual(route[0], project['sectors'][0]['racelines']['1'][0])
        self.assertEqual(route[-1], project['sectors'][-1]['racelines']['1'][-1])
        self.assertTrue(all(a <= b for a, b in zip(sectors, sectors[1:])))
        self.assertEqual(len(route), len(sectors))
        with self.assertRaises(ValueError):
            empty = copy.deepcopy(project)
            empty['sectors'][2]['racelines']['1'].clear()
            primary_route(empty)

    def test_generated_lane_centres_counts_spacing_and_cutoff(self):
        before = copy.deepcopy(self.original)
        project = generate_racelines(self.original, 3.2)
        self.assertEqual(self.original, before)
        layout = project['generated_racelines']
        sector = project['sectors'][10]
        start, stop = layout['start_index'], layout['stop_index']
        reference = before['sectors'][10]['racelines']['1']
        anchor = min(range(len(reference)), key=lambda i:
                     np.hypot(reference[i]['x'] - 63.47,
                              reference[i]['y'] - 146.13))
        self.assertEqual(start, anchor + 1)
        self.assertLessEqual(reference[start - 1]['x'], reference[anchor]['x'])
        self.assertGreater(reference[start]['x'], reference[anchor]['x'])
        self.assertEqual(sector['racelines']['1'], reference)
        self.assertEqual(layout['layout_version'], 2)
        self.assertEqual(layout['settings'], {'raceline_width': 3.2})
        for number in (2, 3):
            lane = sector['racelines'][str(number)]
            self.assertEqual(len(lane), stop - start)
            self.assertTrue(all(p['y'] >= -122.90 for p in lane))
            self.assertTrue(np.allclose(
                np.diff([(p['x'], p['y']) for p in lane], axis=0),
                np.diff([(p['x'], p['y']) for p in reference[start:stop]], axis=0)))
            for original, shifted in zip(reference[start:stop], lane):
                self.assertEqual(shifted['source_id'], original['id'])
                self.assertAlmostEqual(shifted['x'] - original['x'],
                                       (number - 1) * 3.2)
                self.assertEqual((shifted['y'], shifted['z']),
                                 (original['y'], original['z']))
        self.assertLess(reference[stop]['y'], -122.90)
        self.assertEqual(project['sectors'][:10], before['sectors'][:10])
        self.assertEqual(project['sectors'][11:], before['sectors'][11:])

    def test_generated_layout_reload_and_width_change_do_not_accumulate(self):
        first = generate_racelines(self.original, 3.5)
        self.assertEqual(generate_racelines(first, 3.5), first)
        changed = generate_racelines(first, 3.0)
        fresh = generate_racelines(self.original, 3.0)
        for lane_number in ('1', '2', '3'):
            def geometry(project):
                return [(p['x'], p['y'], p['z'], p['speed_kmh'])
                        for p in project['sectors'][10]['racelines'][lane_number]]
            self.assertEqual(geometry(changed), geometry(fresh))
        self.assertEqual(generate_racelines(changed, 3.0), changed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'project.json'
            save_project(path, first)
            self.assertEqual(generate_racelines(load_project(path), 3.5), first)
        for width in (float('nan'), 0, -1, float('inf')):
            with self.assertRaises(ValueError):
                generate_racelines(self.original, width)

    def test_raceline1_trimmed_at_its_lane_drop_opens_but_never_extends(self):
        first = generate_racelines(self.original, 3.5)
        trimmed = copy.deepcopy(first)
        # Saved project since 2026-10-16: the last 139 R1 points before CP11 removed.
        del trimmed['sectors'][10]['racelines']['1'][-139:]
        self.assertEqual(generate_racelines(trimmed, 3.5), trimmed)
        longer = copy.deepcopy(first)
        extra = dict(longer['sectors'][10]['racelines']['1'][-1], id=longer['next_point_id'])
        longer['next_point_id'] += 1
        longer['sectors'][10]['racelines']['1'].append(extra)
        with self.assertRaises(ValueError):
            generate_racelines(longer, 3.5)

    def test_legacy_layout_is_restored_from_csv_without_losing_speed(self):
        legacy = generate_racelines(self.original, 3.2)
        legacy['generated_racelines']['settings']['start_raceline_offset'] = 0.2
        legacy['generated_racelines'].pop('layout_version')
        start = legacy['generated_racelines']['start_index']
        legacy['sectors'][10]['racelines']['1'][start]['x'] += 7.78
        legacy['sectors'][10]['racelines']['1'][start]['speed_kmh'] = 42
        restored = generate_racelines(legacy, 3.2)
        expected = self.original['sectors'][10]['racelines']['1'][start]
        actual = restored['sectors'][10]['racelines']['1'][start]
        self.assertEqual((actual['x'], actual['y'], actual['z']),
                         (expected['x'], expected['y'], expected['z']))
        self.assertEqual(actual['speed_kmh'], 42)
        self.assertNotIn('start_raceline_offset', restored['generated_racelines']['settings'])

    def test_saved_layout_takes_precedence_over_yaml_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'raceline_layout.yaml'
            config.write_text('raceline_width: 3.5\n')
            project = generate_racelines(self.original, 3.2)
            self.assertEqual(raceline_layout_settings(project, config), 3.2)
            self.assertEqual(raceline_layout_settings(project, config, raceline_width=''), 3.2)
            self.assertEqual(raceline_layout_settings(project, config,
                                                     raceline_width=4.0), 4.0)
            self.assertEqual(raceline_layout_settings(self.original, config), 3.5)

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

    def test_range_delete_is_inclusive_one_undo_step_and_protects_raceline1(self):
        editor = ProjectEditor(copy.deepcopy(self.original))
        lane_number = editor.add_lane(3)
        for i in range(25):
            editor.add_point(3, lane_number, float(i), 3.5, 0., 30)
        lane = editor.lane(3, lane_number)
        original = copy.deepcopy(lane)
        first_id, last_id = lane[10]['id'], lane[19]['id']
        self.assertEqual(editor.remove_range(3, lane_number, last_id, first_id), 10)
        self.assertEqual(editor.lane(3, lane_number), original[:10]+original[20:])
        self.assertTrue(editor.undo())
        self.assertEqual(editor.lane(3, lane_number), original)
        r1 = editor.lane(3, 1)
        with self.assertRaises(ValueError):
            editor.remove_range(3, 1, r1[1]['id'], r1[5]['id'])
        # S11 Raceline 1 may be deleted entirely, end points included.
        s11 = editor.lane(11, 1)
        count = len(s11)
        self.assertEqual(editor.remove_range(11, 1, s11[0]['id'], s11[-1]['id']), count)
        self.assertEqual(editor.lane(11, 1), [])
        with self.assertRaises(ValueError):
            editor.remove_range(3, lane_number, first_id, r1[0]['id'])

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

    def test_continuous_points_append_to_existing_raceline3_and_survive_reload(self):
        editor = ProjectEditor(generate_racelines(self.original, 3.5))
        before = copy.deepcopy(editor.project)
        points = editor.lane(11, 3)
        old = copy.deepcopy(points)
        start = (points[-1]['x'] + 1, points[-1]['y'] + 2)
        end = (start[0] + 12, start[1] + 24)
        count = editor.append_continuous_points(11, 3, start, end, 3)
        self.assertEqual(count, 5)
        self.assertEqual(points[:-5], old)
        for index, point in enumerate(points[-5:]):
            self.assertEqual((point['x'], point['y']),
                             (start[0] + index * 3, start[1] + index * 6))
            self.assertEqual(point['speed_kmh'], old[-1]['speed_kmh'])
        self.assertEqual(len({p['id'] for p in points}), len(points))
        expected = copy.deepcopy(before)
        expected['sectors'][10]['racelines']['3'] = copy.deepcopy(points)
        expected['next_point_id'] += 5
        self.assertEqual(editor.project, expected)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'project.json'
            save_project(path, editor.project)
            self.assertEqual(load_project(path), expected)
            self.assertEqual(generate_racelines(load_project(path), 3.5), expected)
        # Another batch can be added beyond the newly created endpoint.
        editor.append_continuous_points(11, 3, end, (end[0] + 10, end[1]), 0)
        self.assertEqual(len(editor.lane(11, 3)), len(old) + 7)
        self.assertTrue(editor.undo())
        self.assertEqual(editor.project, expected)
        self.assertTrue(editor.undo())
        self.assertEqual(editor.project, before)

    def test_continuous_points_empty_lane_and_exact_interior_count(self):
        for interior in (0, 1, 10):
            editor = ProjectEditor(self.original)
            editor.add_lane(1)
            count = editor.append_continuous_points(1, 2, (1, 2), (11, 22), interior)
            points = editor.lane(1, 2)
            self.assertEqual(count, interior + 2)
            self.assertEqual(len(points), interior + 2)
            self.assertEqual((points[0]['x'], points[0]['y']), (1, 2))
            self.assertEqual((points[-1]['x'], points[-1]['y']), (11, 22))
            self.assertEqual(editor.lane(1, 1), self.original['sectors'][0]['racelines']['1'])

    def test_continuous_points_invalid_input_does_not_change_project(self):
        editor = ProjectEditor(self.original)
        before = copy.deepcopy(editor.project)
        for count in (-1, 10001, True, 1.5, '3'):
            with self.assertRaises(ValueError):
                editor.append_continuous_points(1, 1, (0, 0), (1, 1), count)
        for start, end in (((0, 0), (0, 0)), ((0, 0), (float('nan'), 0))):
            with self.assertRaises(ValueError):
                editor.append_continuous_points(1, 1, start, end, 3)
        self.assertEqual(editor.project, before)
        self.assertEqual(editor.history, [])

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
