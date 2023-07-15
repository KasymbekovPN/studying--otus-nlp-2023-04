import unittest

from parameterized import parameterized
from src.hw_003_vector.sentiment.sentiment import Sentiment


class TestCase(unittest.TestCase):

    def test_unknown_value(self):
        self.assertEqual(0, Sentiment.UNKNOWN.value)

    def test_neutral_value(self):
        self.assertEqual(1, Sentiment.NEUTRAL.value)

    def test_positive_value(self):
        self.assertEqual(2, Sentiment.POSITIVE.value)

    def test_negative_value(self):
        self.assertEqual(3, Sentiment.NEGATIVE.value)

    @parameterized.expand([
        ['neutral', Sentiment.NEUTRAL],
        ['neutral', Sentiment.NEUTRAL],
        ['positive', Sentiment.POSITIVE],
        ['POSITIVE', Sentiment.POSITIVE],
        ['negative', Sentiment.NEGATIVE],
        ['NEGATIVE', Sentiment.NEGATIVE],
        ['positive1', Sentiment.UNKNOWN],
        ['POSITIVE2', Sentiment.UNKNOWN],
        ['negative3', Sentiment.UNKNOWN],
        ['NEGATIVE4', Sentiment.UNKNOWN],
        ['abc', Sentiment.UNKNOWN]
    ])
    def test_creation_from_str(self, name: str, expected: Sentiment):
        self.assertEqual(expected, Sentiment.create(name))
    #     self.assertEqual(expected, SemanticEvaluation.create(name))


if __name__ == '__main__':
    unittest.main()
