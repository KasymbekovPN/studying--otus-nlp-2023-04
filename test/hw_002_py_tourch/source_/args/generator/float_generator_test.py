import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.source_.args.generator.float_generator import FloatGenerator


# todo del
class TestCase(unittest.TestCase):

    @parameterized.expand([
        (0.0, 10.0, 1),
        (-10.0, 0.0, 10),
        (100.0, 1000.0, 100)
    ])
    def test_generation(self, min_border: float, max_border: float, quantity: int):
        class TestRange:
            @property
            def min(self):
                return min_border

            @property
            def max(self):
                return max_border

        float_generator = FloatGenerator()
        result = float_generator(TestRange(), quantity)
        self.assertTrue(type(result) is tuple)
        self.assertEqual(quantity, len(result))
        for item in result:
            self.assertTrue(item >= min_border)
            self.assertTrue(item < max_border)


if __name__ == '__main__':
    unittest.main()
