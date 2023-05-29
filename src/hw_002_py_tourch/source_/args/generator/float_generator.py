import random

from src.hw_002_py_tourch.source_.args.generator.base_generator import BaseGenerator


# todo del
class FloatGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__(float)

    def __call__(self, *args, **kwargs) -> tuple:
        float_range = args[0]
        quantity = args[1]
        return tuple(random.uniform(float_range.min, float_range.max) for _ in range(0, quantity))
