
# todo del
class Parts:
    MIN = 1
    MAX = 1_000_000
    DEFAULT_QUANTITY = 10_000

    def __init__(self,
                 train_part: int,
                 test_part: int,
                 validation_part: int):
        self._train_part = self._adjust_or_get_part_arg(train_part)
        self._test_part = self._adjust_or_get_part_arg(test_part)
        self._validation_part = self._adjust_or_get_part_arg(validation_part)

    def __call__(self, *args, **kwargs) -> tuple:
        pass

    def _adjust_or_get_part_arg(self, arg) -> int:
        if arg < self.MIN:
            return self.MIN
        elif arg > self.MAX:
            return self.MAX
        else:
            return arg

    def _adjust_or_get_quantity(self, *args) -> int:
        return args[0] if len(args) > 0 and isinstance(args[0], int) else self.DEFAULT_QUANTITY
