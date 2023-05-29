import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.source_.args.args import Args


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ([], Args.DEFAULT_QUANTITY),
        ([-1], Args.DEFAULT_QUANTITY),
        ([0], Args.DEFAULT_QUANTITY),
        ([1], 1),
        ([10], 10),
        ([10.0], Args.DEFAULT_QUANTITY),
        ([''], Args.DEFAULT_QUANTITY)
    ])
    def test_quantity_checking(self, input_, expected: int):
        args = Args(None, None)
        result = args._check_and_get_quantity(*input_)
        self.assertEqual(expected, result)

    def test_calling_when_types_mismatching(self):
        class TestRange:
            @property
            def type(self):
                return 'type0'

        class TestGenerator:
            @property
            def type(self):
                return 'type1'

        args = Args(TestRange(), TestGenerator())
        with self.assertRaises(Exception) as context:
            _ = args()
        self.assertTrue(args.TYPES_MISMATCH_EXCEPTION_MSG in context.exception.args)

    def test_calling(self):
        expected_result = (0.0, 1.0, 2.0)
        expected_quantity = 3

        class TestRange:
            @property
            def type(self):
                return 'type'

        class TestGenerator:
            @property
            def type(self):
                return 'type'

            def __call__(self, *args, **kwargs):
                self._range = args[0]
                self._quantity = args[1]
                return expected_result

        test_range = TestRange()
        test_generator = TestGenerator()
        args_ = Args(test_range, test_generator)
        result = args_(expected_quantity)

        self.assertEqual(expected_result, result)
        self.assertEqual(expected_quantity, test_generator._quantity)
        self.assertEqual(test_range, test_generator._range)


if __name__ == '__main__':
    unittest.main()
