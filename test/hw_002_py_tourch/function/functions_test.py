import math
import random
import unittest

from src.common.test import repeat
from src.hw_002_py_tourch.function.functions import sin_exp_function


class TestCase(unittest.TestCase):

    @repeat(1_000)
    def test_sin_exp_function(self):
        def test_func(x_, y_):
            return math.sin(x_ + 2 * y_) * math.exp(-1 * ((2 * x_ + y_) ** 2))

        x = random.uniform(-10.0, 10.0)
        y = random.uniform(-10.0, 10.0)
        expected = test_func(x, y)

        self.assertEqual(expected, sin_exp_function(x, y))


if __name__ == '__main__':
    unittest.main()
