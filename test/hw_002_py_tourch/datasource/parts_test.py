import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.datasource.parts import Parts


# todo del
class TestCase(unittest.TestCase):

    @parameterized.expand([
        (Parts.MIN - 1, Parts.MIN),
        (Parts.MIN, Parts.MIN),
        (Parts.MAX, Parts.MAX),
        (Parts.MAX + 1, Parts.MAX)
    ])
    def test_arg_adjusting(self, arg: int, expected: int):
        parts = Parts(0, 0, 0)
        self.assertEqual(expected, parts._adjust_or_get_part_arg(arg))

    @parameterized.expand([
        ([], Parts.DEFAULT_QUANTITY),
        ([12.0], Parts.DEFAULT_QUANTITY),
        (['123'], Parts.DEFAULT_QUANTITY),
        ([1_000, '1'], 1_000),
        ([2_000], 2_000)
    ])
    def test_quantity_adjusting(self, args: list, expected):
        parts = Parts(0, 0, 0)
        self.assertEqual(expected, parts._adjust_or_get_quantity(*args))


if __name__ == '__main__':
    unittest.main()
