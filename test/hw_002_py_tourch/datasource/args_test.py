import unittest

from src.hw_002_py_tourch.datasource.args import create_float_args


class TestCase(unittest.TestCase):
    def test_args_creation(self):
        min_ = -100.0
        max_ = 100.0
        expected_length = 20

        class TestRange:
            @property
            def min(self):
                return min_

            @property
            def max(self):
                return max_

        result = create_float_args(expected_length, TestRange())
        self.assertEqual(expected_length, len(result))
        for item in result:
            self.assertTrue(min_ <= item <= max_)




if __name__ == '__main__':
    unittest.main()
