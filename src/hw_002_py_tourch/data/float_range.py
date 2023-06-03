
class FloatRange:
    DELTA = 0.001
    MIN_SETTER_EXC_MSG = 'Min setter is unsupported'
    MAX_SETTER_EXC_MSG = 'Max setter is unsupported'

    def __init__(self, min_border: float, max_border: float) -> None:
        self._min = min_border
        self._max = max_border if max_border > min_border else min_border + self.DELTA

    @property
    def min(self) -> float:
        return self._min

    @min.setter
    def min(self, value):
        raise Exception(self.MIN_SETTER_EXC_MSG)

    @property
    def max(self) -> float:
        return self._max

    @max.setter
    def max(self, value):
        raise Exception(self.MAX_SETTER_EXC_MSG)