
class Args:
    DEFAULT_QUANTITY = 1
    TYPES_MISMATCH_EXCEPTION_MSG = 'Mismatching of types'

    def __init__(self,
                 arg_range,
                 arg_generator) -> None:
        self._arg_range = arg_range
        self._arg_generator = arg_generator

    def __call__(self, *args, **kwargs) -> tuple:
        if self._arg_generator.type != self._arg_range.type:
            raise Exception(self.TYPES_MISMATCH_EXCEPTION_MSG)

        quantity = self._check_and_get_quantity(*args)
        return self._arg_generator(self._arg_range, quantity)

    def _check_and_get_quantity(self, *args) -> int:
        return args[0]\
            if len(args) > 0 and isinstance(args[0], int) and args[0] > self.DEFAULT_QUANTITY\
            else self.DEFAULT_QUANTITY
