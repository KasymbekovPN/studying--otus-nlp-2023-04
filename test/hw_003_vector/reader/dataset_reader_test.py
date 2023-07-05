import unittest

from parameterized import parameterized
from src.hw_003_vector.reader.dataset_reader import DatasetReader


class TestCase(unittest.TestCase):

    ORIGINAL = [
        {'h0': 'v00', 'h1': 'v10'},
        {'h0': 'v01', 'h1': 'v11'},
        {'h0': 'v02', 'h1': 'v12'}
    ]

    PREPARED = [
        {'h0': '_v00', 'h1': '_v10'},
        {'h0': '_v01', 'h1': '_v11'},
        {'h0': '_v02', 'h1': '_v12'}
    ]

    @parameterized.expand([
        ['bad_path.csv', 'bad_path.csv', None, None],
        ['bad_path.csv', './prepared.csv', None, PREPARED],
        ['./original.csv', 'bad_path.csv', ORIGINAL, None],
        ['./original.csv', './prepared.csv', ORIGINAL, PREPARED],
    ])
    def test_something(self, original_path: str, prepared_path: str, expected_original, expected_prepared):
        reader = DatasetReader()
        reader(path_original=original_path, path_prepared=prepared_path)

        self.assertEqual(expected_original, reader.original)
        self.assertEqual(expected_prepared, reader.prepared)


if __name__ == '__main__':
    unittest.main()
