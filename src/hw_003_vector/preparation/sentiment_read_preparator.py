from src.hw_003_vector.sentiment.sentiment import Sentiment


class SentimentReadPreparator:
    def __init__(self, key='sentiment') -> None:
        self._key = key

    def __call__(self, *args, **kwargs) -> dict:
        datum = kwargs.get('datum')
        datum[self._key] = Sentiment.create(datum[self._key])

        return datum
    