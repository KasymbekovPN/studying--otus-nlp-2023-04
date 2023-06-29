import unittest

from parameterized import parameterized
from src.hw_003_vector.preparation.sentiment_write_preparator import SentimentWritePreparator
from src.hw_003_vector.sentiment.sentiment import Sentiment


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [{'sentiment': Sentiment.UNKNOWN}, {'sentiment': 'unknown'}],
        [{'sentiment': Sentiment.NEUTRAL}, {'sentiment': 'neutral'}],
        [{'sentiment': Sentiment.POSITIVE}, {'sentiment': 'positive'}],
        [{'sentiment': Sentiment.NEGATIVE}, {'sentiment': 'negative'}]
    ])
    def test_sentiment_write_preparation(self, original_datum: dict, expected_datum: dict):
        preparator = SentimentWritePreparator()
        result_datum = preparator(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)


if __name__ == '__main__':
    unittest.main()
