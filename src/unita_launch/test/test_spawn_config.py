import unittest

from unita_visualization.spawn_config import make_start_spawn


class SpawnConfigTests(unittest.TestCase):
    def test_native_schema_and_first_xyz(self):
        config = make_start_spawn(['x,y,z', '-131,-428,28.5', '-131,-427,28.6'])
        self.assertEqual(config, {'m_InitPos': {'x': -131, 'y': -428, 'z': 28.5},
                                  'm_InitRot': {'x': 0, 'y': 0, 'z': 90}})

    def test_duplicate_start_and_comments_do_not_change_heading(self):
        config = make_start_spawn(['# local XYZ', '', '0 0 10', '0 0 11', '1 1 12'])
        self.assertEqual(config['m_InitPos']['z'], 10)
        self.assertEqual(config['m_InitRot']['z'], 45)

    def test_missing_xyz_or_direction_and_invalid_numbers(self):
        for lines in (['0 0', '1 1'], ['0 0 10'], ['0 0 10', '0 0 10'],
                      ['nan 0 10', '1 1 10'], ['0 0 10', '1 inf 10']):
            with self.assertRaises(ValueError):
                make_start_spawn(lines)


if __name__ == '__main__':
    unittest.main()
