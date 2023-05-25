import random


class FloatRandomArgs:
    def __init__(self,
                 min_border: float,
                 max_border: float,
                 quantity=1) -> None:
        self._min_border = min_border
        self._max_border = max_border if max_border > min_border else min_border + 1.0
        self._quantity = 1 if quantity < 1 else quantity

    def __call__(self, *args, **kwargs) -> tuple:
        return tuple([random.uniform(self._min_border, self._max_border) for i in range(0, self._quantity)])
