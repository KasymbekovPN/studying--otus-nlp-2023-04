import torch
import unittest

from src.hw_002_py_tourch.datasource.slicer import slice_two_args_tensor


class TestCase(unittest.TestCase):
    def test_slicing(self):
        tensor = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x_expected = torch.FloatTensor([1.0, 3.0, 5.0])
        y_expected = torch.FloatTensor([2.0, 4.0, 6.0])

        result = slice_two_args_tensor(tensor)
        self.assertEqual(x_expected.numpy().all(), result[0].numpy().all())
        self.assertEqual(y_expected.numpy().all(), result[1].numpy().all())


if __name__ == '__main__':
    unittest.main()
