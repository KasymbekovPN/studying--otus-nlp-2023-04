import random
import unittest
import math

from src.common.test import repeat
from src.hw_002_py_tourch.source.function.float_function import FloatFunction


class TestCase(unittest.TestCase):

    def test_calculation_without_args(self):
        f = FloatFunction()
        with self.assertRaises(Exception) as context:
            f()
        self.assertTrue(f.EXCEPTION_MESSAGE_NO_ONE_ARG in context.exception.args)

    def test_calculation_if_arg_not_callable(self):
        f = FloatFunction()
        with self.assertRaises(Exception) as context:
            f(1)
        self.assertTrue(f.EXCEPTION_MESSAGE_NOT_CALLABLE in context.exception.args)

    @repeat(1_000)
    def test_calculation(self):
        x = random.uniform(-10.0, 10.0)
        y = random.uniform(-10.0, 10.0)

        class TestArgs:
            def __call__(self, *args, **kwargs) -> tuple:
                return x, y

        def check_function(x_arg: float, y_arg: float):
            return math.sin(x_arg + 2 * y_arg) * math.exp(-math.pow((2 * x_arg + y_arg), 2))

        f = FloatFunction()
        result = f(TestArgs())
        self.assertEqual(check_function(x, y), result)


if __name__ == '__main__':
    unittest.main()
