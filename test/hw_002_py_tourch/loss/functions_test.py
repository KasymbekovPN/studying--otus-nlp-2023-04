import unittest
import random

from src.common.test import repeat
from src.hw_002_py_tourch.loss.functions import compute_mse_loss


class TestCase(unittest.TestCase):

    @repeat(1_000)
    def test_mse_loss_computing(self):
        class TestArg:
            def __init__(self, value: int) -> None:
                self._value = value

            def __pow__(self, power, modulo=None):
                return TestArg(self._value ** power)

            def __sub__(self, other):
                return TestArg(self._value - other._value)

            def mean(self):
                return self._value

        prediction = random.randint(1, 10)
        target = random.randint(1, 10)
        expected = (prediction - target) ** 2

        result = compute_mse_loss(TestArg(prediction), TestArg(target))
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
