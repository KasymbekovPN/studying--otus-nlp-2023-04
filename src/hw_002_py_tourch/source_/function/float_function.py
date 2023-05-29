import math


# todo del
class FloatFunction:
    ARGS_QUANTITY = 2
    EXCEPTION_MESSAGE_NO_ONE_ARG = 'No one args-generator'
    EXCEPTION_MESSAGE_NOT_CALLABLE = 'Args-generator is not callable'

    def __call__(self, *args, **kwargs) -> float:
        if len(args) == 0:
            raise Exception(self.EXCEPTION_MESSAGE_NO_ONE_ARG)
        args_ = args[0]
        if not callable(args_):
            raise Exception(self.EXCEPTION_MESSAGE_NOT_CALLABLE)
        x, y = args_(self.ARGS_QUANTITY)
        return math.sin(x + 2*y) * math.exp(-math.pow((2*x + y), 2))
