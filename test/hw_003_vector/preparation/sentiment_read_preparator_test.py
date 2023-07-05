import unittest

from parameterized import parameterized
from src.hw_003_vector.preparation.sentiment_read_preparator import SentimentReadPreparator
from src.hw_003_vector.sentiment.sentiment import Sentiment


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [{'sentiment': 'abcde24'}, {'sentiment': Sentiment.UNKNOWN}],
        [{'sentiment': 'unknown'}, {'sentiment': Sentiment.UNKNOWN}],
        [{'sentiment': 'UNKNOWN'}, {'sentiment': Sentiment.UNKNOWN}],
        [{'sentiment': 'neutral'}, {'sentiment': Sentiment.NEUTRAL}],
        [{'sentiment': 'NEUTRAL'}, {'sentiment': Sentiment.NEUTRAL}],
        [{'sentiment': 'positive'}, {'sentiment': Sentiment.POSITIVE}],
        [{'sentiment': 'POSITIVE'}, {'sentiment': Sentiment.POSITIVE}],
        [{'sentiment': 'negative'}, {'sentiment': Sentiment.NEGATIVE}],
        [{'sentiment': 'NEGATIVE'}, {'sentiment': Sentiment.NEGATIVE}]
    ])
    def test_sentiment_read_preparation(self, original_datum: dict, expected_datum: dict):
        preparator = SentimentReadPreparator()
        result_datum = preparator(datum=original_datum)
        self.assertEqual(expected_datum, result_datum)


if __name__ == '__main__':
    unittest.main()
