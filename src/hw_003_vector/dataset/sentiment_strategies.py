from src.hw_003_vector.sentiment.sentiment import Sentiment


def convert_sentiment_to_int(sentiment: Sentiment) -> int:
    return sentiment.value


def convert_sentiment_to_str(sentiment: Sentiment) -> str:
    return sentiment.name.lower()
