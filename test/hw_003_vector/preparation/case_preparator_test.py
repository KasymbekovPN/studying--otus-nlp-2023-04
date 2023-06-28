import unittest

from parameterized import parameterized
from src.hw_003_vector.preparation.case_preparator import CasePreparator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [True, 'aBcDe', 'ABCDE'],
        [False, 'aBcDe', 'abcde']
    ])
    def test_case_preparation(self, up: bool, original_text: str, expected_text: str):
        preparator = CasePreparator(up)
        result_text = preparator(text=original_text)
        self.assertEqual(expected_text, result_text)


if __name__ == '__main__':
    unittest.main()
