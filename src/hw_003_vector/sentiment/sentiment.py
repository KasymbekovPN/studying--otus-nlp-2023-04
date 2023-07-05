from enum import Enum


class Sentiment(Enum):
    UNKNOWN = 0
    NEUTRAL = 1
    POSITIVE = 2
    NEGATIVE = 3

    @classmethod
    def create(cls, name: str) -> 'Sentiment':
        name = name.lower()
        if name == 'neutral':
            return Sentiment.NEUTRAL
        if name == 'positive':
            return Sentiment.POSITIVE
        if name == 'negative':
            return Sentiment.NEGATIVE
        return Sentiment.UNKNOWN
