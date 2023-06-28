
class CasePreparator:
    def __init__(self, up=False):
        self._up = up

    def __call__(self, *args, **kwargs):
        original_text = kwargs.get('text')
        return original_text.upper() if self._up else original_text.lower()
    