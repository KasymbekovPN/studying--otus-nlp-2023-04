import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.data.linear_ds import ComplexLinearDS


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

    def test_tensor_creation(self):
        expected_len = 10
        ds = ComplexLinearDS()
        tensor = ds._create_tensor(expected_len)

        self.assertEqual(expected_len, len(tensor))
        for item in tensor:
            print(item, ' ', type(item))
            self.assertTrue(-10.0 <= item <= 10.0)

    def test_calling(self):
        train_len = 10
        test_len = 11
        val_len = 12
        kwargs = {
            'train': train_len,
            'test': test_len,
            'val': val_len
        }
        ds = ComplexLinearDS()
        result = ds(**kwargs)

        self.assertEqual(3, len(result))


if __name__ == '__main__':
    unittest.main()
