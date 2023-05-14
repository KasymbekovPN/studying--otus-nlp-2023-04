from parameterized import parameterized
import unittest

from src.hw_001_data_parsing.datasource.ds_feed_page_links import calculate_pack_ranges, make_replacing, DSFeedPageLinks
from src.hw_001_data_parsing.configurator.configurator import Configurator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        (3, 8, [[0, 3]]),
        (17, 7, [[0, 7], [7, 14], [14, 17]]),
        (8, 4, [[0, 4], [4, 8]])
    ])
    def test_range_calculation(self, page_amount: int, thread_amount: int, expected: list):
        configurator = Configurator()
        configurator.thread_amount = thread_amount
        configurator.max_feed_page_amount = page_amount

        result = calculate_pack_ranges(configurator)
        self.assertEqual(isinstance(result, list), True)
        self.assertEqual(result, expected)

    @parameterized.expand([
        ('https://habr.com', 123, 'https://habr.com'),
        ('https://habr.com/ru/all/page{i}/', 42, 'https://habr.com/ru/all/page43/')
    ])
    def test_replacing(self, template: str, idx: int, expected):
        result = make_replacing(template, idx)
        self.assertEqual(True, isinstance(result, str))
        self.assertEqual(expected, result)

    def test_creation(self):
        def calculate_test_pack_ranges(configurator: 'Configurator') -> list:
            return [
                [0, 7],
                [7, 10]
            ]

        def make_test_replacing(template: str, idx: int) -> str:
            return 'some.str.{i}'.replace('{i}', str(idx))

        expected_pack_0 = [make_test_replacing('', i) for i in range(0, 7)]
        expected_pack_1 = [make_test_replacing('', i) for i in range(7, 10)]

        ds = DSFeedPageLinks(None, calculate_test_pack_ranges, make_test_replacing)
        pack = ds.link_pack

        self.assertEqual(2, len(pack))

        result_0 = [l for l in pack[0]]
        result_1 = [l for l in pack[1]]

        self.assertEqual(expected_pack_0, result_0)
        self.assertEqual(expected_pack_1, result_1)


if __name__ == '__main__':
    unittest.main()
