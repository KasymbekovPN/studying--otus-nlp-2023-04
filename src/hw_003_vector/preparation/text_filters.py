import re
from typing import Pattern, AnyStr


class RegFilter:
    def __init__(self, pattern: Pattern[AnyStr], new_sub=' '):
        self._pattern = pattern
        self._num_sub = new_sub

    def filter(self, text: str) -> str:
        return self._pattern.sub(self._num_sub, text)


class TagsFilter(RegFilter):
    def __init__(self):
        super().__init__(re.compile(r'<[a-z][\w=" -]+/?>'))


class PunctuationFilter(RegFilter):
    def __init__(self):
        super().__init__(re.compile(r'[.,!?*_)(]+'))


class NormSpaceFilter(RegFilter):
    def __init__(self):
        super().__init__(re.compile(r'\s+'))
