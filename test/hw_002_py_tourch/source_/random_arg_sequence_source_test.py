import unittest

from src.common.test import repeat


# todo del
class TestCase(unittest.TestCase):

    @repeat(10)
    def test_sequence_creation(self):
        args = (1.0, 2.0)
        print('------------------')


if __name__ == '__main__':
    unittest.main()
