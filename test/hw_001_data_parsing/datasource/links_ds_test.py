from parameterized import parameterized
import unittest

from src.hw_001_data_parsing.datasource.links_ds import compute_pack_ranges, LinksDS


class TestCase(unittest.TestCase):

    @parameterized.expand([
        (3, 8, [[0, 3]]),
        (17, 7, [[0, 7], [7, 14], [14, 17]]),
        (8, 4, [[0, 4], [4, 8]])
    ])
    def test_range_calculation(self, links_quantity: int, threads_quantity: int, expected: list):
        result = compute_pack_ranges(threads_quantity, links_quantity)
        self.assertEqual(result, expected)

    def test_links_ds_creation(self):
        links_quantity = 13
        threads_quantity = 4
        links = [f'https://link{i}.org' for i in range(0, links_quantity)]

        class TestLinksCreator:
            def __init__(self, ls: list): self._links = ls
            def __call__(self, *args, **kwargs): return self._links

        def test_compute_pack_ranges(tq: int, lq: int) -> list:
            return [[0, 4], [4, 8], [8, 12], [12, 13]]

        expected_result = [
            ['https://link0.org',
             'https://link1.org',
             'https://link2.org',
             'https://link3.org'],
            ['https://link4.org',
             'https://link5.org',
             'https://link6.org',
             'https://link7.org'],
            ['https://link8.org',
             'https://link9.org',
             'https://link10.org',
             'https://link11.org'],
            ['https://link12.org']
        ]

        ds = LinksDS(threads_quantity, TestLinksCreator(links), test_compute_pack_ranges)
        self.assertEqual(expected_result, ds.link_packs)


if __name__ == '__main__':
    unittest.main()
