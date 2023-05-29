import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.datasource.linear_ds import ComplexLinearDS


class TestCase(unittest.TestCase):

    TEST_KEY = 'some.key'

    @parameterized.expand([
        ({}, ComplexLinearDS.DEFAULT_LEN),
        ({TEST_KEY: 'hello'}, ComplexLinearDS.DEFAULT_LEN),
        ({TEST_KEY: -1}, ComplexLinearDS.DEFAULT_LEN),
        ({TEST_KEY: ComplexLinearDS.MIN_LEN}, ComplexLinearDS.MIN_LEN),
        ({TEST_KEY: ComplexLinearDS.MAX_LEN}, ComplexLinearDS.MAX_LEN)
    ])
    def test_len_adjusting(self, kwargs: dict, expected: int):
        ds = ComplexLinearDS()
        self.assertEqual(expected, ds._adjust_or_get_len(self.TEST_KEY, **kwargs))

    # todo  _adjust_range -- check it


if __name__ == '__main__':
    unittest.main()
