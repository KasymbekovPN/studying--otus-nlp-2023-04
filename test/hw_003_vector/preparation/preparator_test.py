import unittest

from src.hw_003_vector.preparation.preparator import Preparator


class TestCase(unittest.TestCase):
    def test_something(self):
        def test_filter(text: str):
            return text + 'b'

        class NotCallableClass:
            pass

        class CallableClass:
            def __call__(self, *args, **kwargs):
                return args[0] + 'c'

        text = 'a'
        expected_text = 'abc'

        preparator = Preparator(test_filter, NotCallableClass(), CallableClass())
        result_text = preparator(text)

        self.assertEqual(expected_text, result_text)


if __name__ == '__main__':
    unittest.main()
