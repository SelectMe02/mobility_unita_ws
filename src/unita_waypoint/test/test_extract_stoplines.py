import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/extract_stoplines.py'
spec = importlib.util.spec_from_file_location('extract_stoplines', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExtractStoplinesTests(unittest.TestCase):
    def test_filters_signal_nodes_and_counts_each_link_direction(self):
        nodes = {
            'n1': {'idx': 'n1', 'point': [10., 2., 0.],
                   'on_stop_line': True, 'traffic_light_id': 't1'},
            'n2': {'idx': 'n2', 'point': [10., 5., 0.],
                   'on_stop_line': True, 'traffic_light_id': 't1'},
            'n3': {'idx': 'n3', 'point': [0., 2., 0.],
                   'on_stop_line': True, 'traffic_light_id': None},
        }
        links = {
            'a': {'idx': 'a', 'from_node_idx': 'n3', 'to_node_idx': 'n1'},
            'b': {'idx': 'b', 'from_node_idx': 'n1', 'to_node_idx': 'n2'},
            'c': {'idx': 'c', 'from_node_idx': 'n1', 'to_node_idx': 'n3'},
        }
        result = module.extract(nodes, links)
        self.assertEqual(result, [
            {'idx': 'n1', 'x': 10., 'y': 2., 'traffic_light_id': 't1',
             'from_link_idxs': ['b', 'c'], 'to_link_idxs': ['a']},
            {'idx': 'n2', 'x': 10., 'y': 5., 'traffic_light_id': 't1',
             'from_link_idxs': [], 'to_link_idxs': ['b']},
        ])
        self.assertEqual(module.statistics(result), {
            'stopline_nodes': 2, 'from_link_references': 2,
            'to_link_references': 2, 'nodes_with_from_links': 1,
            'nodes_with_to_links': 2, 'nodes_with_both': 1,
            'nodes_with_neither': 0,
        })


if __name__ == '__main__':
    unittest.main()
