import unittest

from src.hw_003_vector.preparation.preparation import Preparation


class TestCase(unittest.TestCase):
    def test_something(self):
        def test_filter(t: str):
            return t + 'b'

        class NotCallableClass:
            pass

        class CallableClass:
            def __call__(self, *args, **kwargs):
                return args[0] + 'c'

        text = 'a'
        expected_text = 'abc'

        preparator = Preparation(test_filter, NotCallableClass(), CallableClass())
        result_text = preparator(text)

        self.assertEqual(expected_text, result_text)


if __name__ == '__main__':
    unittest.main()
