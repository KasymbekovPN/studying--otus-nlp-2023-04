import re
from typing import Pattern, AnyStr


class RegexPreparator:
    def __init__(self, pattern: Pattern[AnyStr], new_sub=' '):
        self._pattern = pattern
        self._num_sub = new_sub

    def __call__(self, *args, **kwargs):
        return self._pattern.sub(self._num_sub, kwargs.get('text'))


class HtmlTagsPreparator(RegexPreparator):
    def __init__(self):
        super().__init__(re.compile(r'<[a-z][\w=" -]+/?>'))


class PunctuationPreparator(RegexPreparator):
    def __init__(self):
        super().__init__(re.compile(r'[.,!?*_)(]+'))


class SpacePreparator(RegexPreparator):
    def __init__(self):
        super().__init__(re.compile(r'\s+'))
