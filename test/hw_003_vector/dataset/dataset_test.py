import unittest

from parameterized import parameterized
from src.hw_003_vector.dataset.dataset import Dataset


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ['value1', 'value1'],
        [123, 123],
        [None, None]
    ])
    def test_default_strategy_running(self, original_value, expected_value):
        result_value = Dataset.run_default_strategy(original_value)
        self.assertEqual(expected_value, result_value)

    @parameterized.expand([
        [None, set()],
        [123, set()],
        ['', set()],
        [set(), set()],
        [{123, None, 56.9}, set()],
        [{'', None, 123}, set()],
        [{'header0', '', 'header1', 123, 'header3', None}, {'header0', 'header1', 'header3'}]
    ])
    def test_headers_preparation(self, original: set, expected: str):
        result = Dataset._prepare_headers(original)
        self.assertEqual(expected, result)

    def test_data_preparation(self):
        headers = {'h0', 'h1', 'h2'}
        raw_data = [
            {},
            {'h0': 1_000, 'h1': 2_000, 'h2': 3_000},
            {'h0': 1_001, 'h1': 2_001},
            {'h0': 1_002,'h2': 3_002},
            {'h1': 2_003, 'h2': 3_003},
            {'h0': 1_004, 'h1': 2_004, 'h2': 3_004},
            {'h0': 1_005, 'h1': 2_005, 'h2': 3_005},
            {'h0': 1_006, 'h1': 2_006, 'h2': 3_006}
        ]
        expected = {
            'h0': [1_000, 1_004, 1_005, 1_006],
            'h1': [2_000, 2_004, 2_005, 2_006],
            'h2': [3_000, 3_004, 3_005, 3_006]
        }

        result = Dataset._prepare_data(raw_data, headers)
        self.assertEqual(expected, result)

    def test_strategies_preparation(self):
        def run_test_strategy(original):
            pass

        headers = {'h0', 'h1'}
        result = Dataset._prepare_strategies(headers, strategy_h0=run_test_strategy, strategy_h2=run_test_strategy)

        expected = {
            'h0': Dataset.run_default_strategy,
            'h1': Dataset.run_default_strategy,
            'h2': run_test_strategy
        }
        self.assertEqual(result, expected)

    def test_getting(self):
        headers = {'h1', 'h2'}
        raw_data = [
            {'h1': 100, 'h2': 200},
            {'h1': 101, 'h2': 201}
        ]
        expected = {
            'h1': [100, 101],
            'h2': [200, 201]
        }

        dataset = Dataset(raw_data, headers)
        result = dataset.get()

        self.assertEqual(expected, result)

    def test_column_getting(self):
        headers = {'h1'}
        size = 10
        raw_data = [{'h1': i} for i in range(size)]
        expected_h1 = [i for i in range(size)]
        expected_inc_h1 = [i+1 for i in range(size)]

        def run_inc_strategy(original):
            return original + 1

        dataset = Dataset(raw_data, headers, strategy_inc=run_inc_strategy)
        result_h1 = dataset.get_column('h1')
        result_inc = dataset.get_column('h1', 'inc')

        self.assertEqual(expected_h1, result_h1)
        self.assertEqual(expected_inc_h1, result_inc)


if __name__ == '__main__':
    unittest.main()
