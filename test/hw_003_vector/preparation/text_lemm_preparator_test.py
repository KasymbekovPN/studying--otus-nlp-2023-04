import unittest

from parameterized import parameterized
from src.hw_003_vector.preparation.text_lemm_preparator import TextLemmPreparator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        ['i had written tests', 'i have write test'],
        ['i believed i can fly', 'i believe i can fly']
    ])
    def test_lemmatization(self, original_text, expected_text):
        lem = TextLemmPreparator()
        result_text = lem(text=original_text)
        self.assertEqual(expected_text, result_text)


if __name__ == '__main__':
    unittest.main()
