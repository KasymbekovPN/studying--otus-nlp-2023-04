import unittest

from parameterized import parameterized
from src.hw_002_py_tourch.source_.args.generator.base_generator import BaseGenerator

# todo del

class TestCase(unittest.TestCase):
    @parameterized.expand([
        (float, float),
        (str, str),
        (int, int),
        (1, 1),
        (1.0, 1.0),
        ('some.type', 'some.type')
    ])
    def test_type_getting(self, init_type, expected_type):
        base_generator = BaseGenerator(init_type)
        self.assertEqual(expected_type, base_generator.type)

    def test_type_setting(self):
        base_generator = BaseGenerator(float)
        with self.assertRaises(Exception) as context:
            base_generator.type = 123
        self.assertTrue(base_generator.TYPE_SETTER_EXC_MSG in context.exception.args)


if __name__ == '__main__':
    unittest.main()
