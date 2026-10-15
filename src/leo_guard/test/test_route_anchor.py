"""Surveyed route features follow the current Map Tuner route by map anchor."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leo_guard.route_file import anchor_on_route, load_lane_changes, load_survey


class RouteAnchorTests(unittest.TestCase):
    # Out along y=0, back along y=5: the same x appears on both passes.
    ROUTE = np.array([[0., 0.], [50., 0.], [100., 0.], [100., 5.], [50., 5.], [0., 5.]])

    def test_anchor_resolves_the_pass_with_matching_heading(self):
        index, s = anchor_on_route(self.ROUTE, dict(x=60., y=2., yaw=0.))
        self.assertEqual(index, 1); self.assertAlmostEqual(s, 60.)
        index, s = anchor_on_route(self.ROUTE, dict(x=60., y=3., yaw=math.pi))
        self.assertEqual(index, 3); self.assertAlmostEqual(s, 145.)

    def test_anchor_far_from_route_is_rejected(self):
        with self.assertRaises(ValueError):
            anchor_on_route(self.ROUTE, dict(x=60., y=20., yaw=0.))

    def test_route_edit_moves_index_and_s_with_the_anchor(self):
        anchor = dict(x=80., y=0., yaw=0.)
        denser = np.vstack([[0., 0.], [25., 0.], self.ROUTE[1:]])
        self.assertEqual(anchor_on_route(self.ROUTE, anchor), (1, 80.))
        self.assertEqual(anchor_on_route(denser, anchor), (2, 80.))

    def test_survey_and_lane_changes_get_route_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            survey = Path(folder) / 'survey.json'
            survey.write_text(json.dumps(dict(intersections=[
                dict(id='a', clearance_m=10., anchor=dict(x=30., y=0., yaw=0.))])))
            changes = Path(folder) / 'changes.json'
            changes.write_text(json.dumps(dict(lane_changes=[dict(
                start=dict(x=10., y=0., yaw=0.), end=dict(x=40., y=0., yaw=0.),
                side='left', offset_m=3.5)])))
            stop = load_survey(str(survey), self.ROUTE)[0]
            self.assertEqual((stop['s'], stop['waypoint_index'], stop['id']), (30., 0, 'a'))
            change = load_lane_changes(str(changes), self.ROUTE)[0]
            self.assertEqual((change['start_s'], change['end_s'], change['offset_m']), (10., 40., 3.5))
            self.assertNotIn('start', change)


if __name__ == '__main__':
    unittest.main()
