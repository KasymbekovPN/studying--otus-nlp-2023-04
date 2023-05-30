import torch
import unittest

from src.hw_002_py_tourch.function.functions import sin_exp_function


class TestCase(unittest.TestCase):

    def test_sin_exp_function(self):
        args = torch.FloatTensor([
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2]
        ])
        x = torch.FloatTensor([1.0, 1.1, 1.2])
        y = torch.FloatTensor([2.0, 2.1, 2.2])
        expected = torch.sin(x + 2*y) * torch.exp(-1 * ((2*x + y) ** 2))

        result = sin_exp_function(args)
        self.assertEqual(expected.numpy().all(), result.numpy().all())


if __name__ == '__main__':
    unittest.main()
