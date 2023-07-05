import unittest

from parameterized import parameterized
from src.hw_003_vector.preparation.text_lemm_preparator import TextLemmPreparator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [{'review': 'i had written tests'}, {'review': 'i have write test'}],
        [{'review': 'i believed i can fly'}, {'review': 'i believe i can fly'}]
    ])
    def test_lemmatization(self, original_datum: dict, expected_datum: dict):
        lem = TextLemmPreparator()
        result_datum = lem(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)


if __name__ == '__main__':
    unittest.main()
