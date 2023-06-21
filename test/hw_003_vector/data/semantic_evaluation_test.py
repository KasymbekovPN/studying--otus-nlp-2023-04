import unittest

from parameterized import parameterized
from src.hw_003_vector.data.semantic_evaluation import SemanticEvaluation


class TestCase(unittest.TestCase):

    def test_unknown_value(self):
        self.assertEqual(0, SemanticEvaluation.UNKNOWN.value)

    def test_positive_value(self):
        self.assertEqual(1, SemanticEvaluation.POSITIVE.value)

    def test_negative_value(self):
        self.assertEqual(2, SemanticEvaluation.NEGATIVE.value)

    @parameterized.expand([
        ['positive', SemanticEvaluation.POSITIVE],
        ['POSITIVE', SemanticEvaluation.POSITIVE],
        ['negative', SemanticEvaluation.NEGATIVE],
        ['NEGATIVE', SemanticEvaluation.NEGATIVE],
        ['positive1', SemanticEvaluation.UNKNOWN],
        ['POSITIVE2', SemanticEvaluation.UNKNOWN],
        ['negative3', SemanticEvaluation.UNKNOWN],
        ['NEGATIVE4', SemanticEvaluation.UNKNOWN],
        ['abc', SemanticEvaluation.UNKNOWN]
    ])
    def test_creation_from_str(self, name: str, expected: SemanticEvaluation):
        self.assertEqual(expected, SemanticEvaluation.create(name))


if __name__ == '__main__':
    unittest.main()
