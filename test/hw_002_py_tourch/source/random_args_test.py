import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.source.random_args import FloatRandomArgs


class TestCase(unittest.TestCase):

    @parameterized.expand([
        (0.0, -1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 10.0, 0.0, 10.0)
    ])
    def test_min_max_setting(self, original_min: float, original_max: float, expected_min: float, expected_max: float):
        random_args = FloatRandomArgs(original_min, original_max)
        self.assertEqual(expected_min, random_args._min_border)
        self.assertEqual(expected_max, random_args._max_border)

    @parameterized.expand([
        (-1, 1),
        (0, 1),
        (1, 1),
        (2, 2)
    ])
    def test_quantity_setting(self, original_quantity: int, expected_quantity: int):
        random_args = FloatRandomArgs(0.0, 1.0, original_quantity)
        self.assertEqual(expected_quantity, random_args._quantity)

    @parameterized.expand([
        (0.0, 10.0, 1),
        (-3.0, 0.0, 5),
        (-10.0, 10.0, 10)
    ])
    def test_creation(self, min_border: float, max_border: float, quantity: int):
        random_args = FloatRandomArgs(min_border, max_border, quantity)
        result = random_args()
        self.assertEqual(quantity, len(result))
        for arg in result:
            self.assertEqual(True, arg >= min_border)
            self.assertEqual(True, arg <= max_border)


if __name__ == '__main__':
    unittest.main()
