import unittest
import torch

from src.hw_002_py_tourch.data.array_processor import group_line_float_tensors, Message


class TestCase(unittest.TestCase):
    def test_float_line_tensors_grouping__if_args_empty(self):
        with self.assertRaises(Exception) as context:
            group_line_float_tensors()
        self.assertTrue(Message.EMPTY_ARGS.value in context.exception.args)

    def test_float_line_tensors_grouping__if_disallowed_arg(self):
        with self.assertRaises(Exception) as context:
            group_line_float_tensors(1)
        self.assertTrue(Message.DISALLOWED_ARG.value in context.exception.args)

    def test_float_line_tensors_grouping__if_arg_has_bad_dimension(self):
        x = torch.FloatTensor([[2, 2], [3, 3]])
        with self.assertRaises(Exception) as context:
            group_line_float_tensors(x)
        self.assertTrue(Message.BAD_DIM.value in context.exception.args)

    def test_float_line_tensors_grouping__if_arg_empty(self):
        x = torch.FloatTensor()
        with self.assertRaises(Exception) as context:
            group_line_float_tensors(x)
        self.assertTrue(Message.EMPTY_TENSORS.value in context.exception.args)

    def test_float_line_tensors_grouping__if_difference_lens(self):
        x1 = torch.FloatTensor([1])
        x2 = torch.FloatTensor([2, 2])
        x3 = torch.FloatTensor([3, 3, 3])
        with self.assertRaises(Exception) as context:
            group_line_float_tensors(x1, x2, x3)
        self.assertTrue(Message.DIFF_LENS.value in context.exception.args)

    def test_float_line_tensors_grouping(self):
        x = torch.FloatTensor([1.0, 1.1, 1.2])
        y = torch.FloatTensor([2.0, 2.1, 2.2])
        expected = torch.FloatTensor([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]])

        result = group_line_float_tensors(x, y)
        self.assertTrue(torch.equal(expected, result))


if __name__ == '__main__':
    unittest.main()
