
class CasePreparator:
    def __init__(self, up=False, text_key='review'):
        self._up = up
        self._text_key = text_key

    def __call__(self, *args, **kwargs):
        datum = kwargs.get('datum')
        datum[self._text_key] = datum[self._text_key].upper() if self._up else datum[self._text_key].lower()

        return datum
