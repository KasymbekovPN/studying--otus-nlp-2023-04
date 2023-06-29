
class SentimentWritePreparator:
    def __init__(self, key='sentiment'):
        self._key = key

    def __call__(self, *args, **kwargs):
        datum = kwargs.get('datum')
        datum[self._key] = datum[self._key].name.lower()

        return datum
