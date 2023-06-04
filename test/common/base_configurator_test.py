import unittest

from parameterized import parameterized
from src.common.base_configurator import BaseConfigurator


class TestCase(unittest.TestCase):

    BAD_KEY = 'bad.key'
    KEY = 'some.key'
    VALUE = 'some.value'

    class TestConfigurator(BaseConfigurator):
        def __init__(self):
            params = {TestCase.KEY: TestCase.VALUE}
            super().__init__(params)

    @parameterized.expand([
        ([], None),
        ([BAD_KEY], None),
        ([KEY], VALUE)
    ])
    def test_base_configurator(self, args: list, expected_value):
        conf = TestCase.TestConfigurator()
        self.assertEqual(expected_value, conf(*args))


if __name__ == '__main__':
    unittest.main()
