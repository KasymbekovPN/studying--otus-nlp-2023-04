import numpy as np

from src.hw_002_py_tourch.data.float_range import FloatRange


def create_random_float_args(length: int, range_=FloatRange(-10.0, 10.0)) -> 'ndarray':
    return np.sort(np.random.uniform(range_.min, range_.max, [length]))

