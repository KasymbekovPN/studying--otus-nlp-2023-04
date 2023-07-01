import unittest

from parameterized import parameterized
from src.hw_003_vector.sentiment.sentiment import Sentiment
from src.hw_003_vector.dataset.sentiment_strategies import convert_sentiment_to_int, convert_sentiment_to_str


class TestCase(unittest.TestCase):

    @parameterized.expand([
        [Sentiment.UNKNOWN, Sentiment.UNKNOWN.value],
        [Sentiment.NEUTRAL, Sentiment.NEUTRAL.value],
        [Sentiment.POSITIVE, Sentiment.POSITIVE.value],
        [Sentiment.NEGATIVE, Sentiment.NEGATIVE.value]
    ])
    def test_sentiment_to_int_conversion(self, original_value: Sentiment, expected_value: int):
        result = convert_sentiment_to_int(original_value)
        self.assertEqual(expected_value, result)

    @parameterized.expand([
        [Sentiment.UNKNOWN, Sentiment.UNKNOWN.name.lower()],
        [Sentiment.NEUTRAL, Sentiment.NEUTRAL.name.lower()],
        [Sentiment.POSITIVE, Sentiment.POSITIVE.name.lower()],
        [Sentiment.NEGATIVE, Sentiment.NEGATIVE.name.lower()]
    ])
    def test_sentiment_to_str_conversion(self, original_value: Sentiment, expected_value: str):
        result = convert_sentiment_to_str(original_value)
        self.assertEqual(expected_value, result)


if __name__ == '__main__':
    unittest.main()
