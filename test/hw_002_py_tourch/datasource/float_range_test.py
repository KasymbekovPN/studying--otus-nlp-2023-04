import random
import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.datasource.float_range import FloatRange


class TestCase(unittest.TestCase):

    @parameterized.expand([
        (10.0, -10.0, 10.0, 10.0 + FloatRange.DELTA),
        (10.0, 10.0, 10.0, 10.0 + FloatRange.DELTA),
        (10.0, 110.0, 10.0, 110.0)
    ])
    def test_float_range_initiation(self, init_min: float, init_max: float, expected_min: float, expected_max: float):
        float_range = FloatRange(init_min, init_max)
        self.assertEqual(expected_min, float_range._min)
        self.assertEqual(expected_max, float_range._max)

    def test_min_getter(self):
        expected_min = random.uniform(10.0, 20.0)
        float_range = FloatRange(expected_min, expected_min + 1.0)
        self.assertEqual(expected_min, float_range.min)

    def test_min_setter(self):
        expected_min = random.uniform(10.0, 20.0)
        float_range = FloatRange(expected_min, expected_min + 1.0)
        with self.assertRaises(Exception) as content:
            float_range.min = 0.0
        self.assertTrue(float_range.MIN_SETTER_EXC_MSG in content.exception.args)

    def test_max_getter(self):
        expected_max = random.uniform(10.0, 20.0)
        float_range = FloatRange(expected_max - 1.0, expected_max)
        self.assertEqual(expected_max, float_range.max)

    def test_max_setter(self):
        expected_max = random.uniform(10.0, 20.0)
        float_range = FloatRange(expected_max - 1.0, expected_max)
        with self.assertRaises(Exception) as content:
            float_range.max = 0.0
        self.assertTrue(float_range.MAX_SETTER_EXC_MSG in content.exception.args)


if __name__ == '__main__':
    unittest.main()
