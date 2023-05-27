
class Args:
    DEFAULT_AND_MIN_QUANTITY = 1

    def __init__(self,
                 arg_range) -> None:
        self._arg_range = arg_range

    def __call__(self, *args, **kwargs) -> tuple:
        pass
        # return tuple([random.uniform(self._min_border, self._max_border) for i in range(0, self._quantity)])

    def _check_and_get_quantity(self, *args) -> int:
        pass
