import unittest

from parameterized import parameterized
from src.hw_003_vector.preparation.case_preparator import CasePreparator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [True, {'review': 'aBcDe'}, {'review': 'ABCDE'}],
        [False, {'review': 'aBcDe'}, {'review': 'abcde'}]
    ])
    def test_case_preparation(self, up: bool, original_datum: dict, expected_datum: dict):
        preparator = CasePreparator(up)
        result_datum = preparator(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)


if __name__ == '__main__':
    unittest.main()
