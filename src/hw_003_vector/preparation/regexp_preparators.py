import re
from typing import Pattern, AnyStr


class RegexPreparator:
    def __init__(self, pattern: Pattern[AnyStr], new_sub=' ', text_key='review'):
        self._pattern = pattern
        self._new_sub = new_sub
        self._text_key = text_key

    def __call__(self, *args, **kwargs):
        datum = kwargs.get('datum')
        datum[self._text_key] = self._pattern.sub(self._new_sub, datum[self._text_key])

        return datum


class HtmlTagsPreparator(RegexPreparator):
    def __init__(self):
        super().__init__(re.compile(r'<[a-z][\w=" -]+/?>'))


class PunctuationPreparator(RegexPreparator):
    def __init__(self):
        super().__init__(re.compile(r'[.,!?*_)(]+'))


class SpacePreparator(RegexPreparator):
    def __init__(self):
        super().__init__(re.compile(r'\s+'))
