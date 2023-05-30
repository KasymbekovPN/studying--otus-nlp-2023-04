import torch

from src.hw_002_py_tourch.datasource.float_range import FloatRange


class ComplexLinearDS:
    MIN_LEN = 1
    MAX_LEN = 1_000_000
    DEFAULT_LEN = 10_000
    KEYS = ('train_len', 'test_len', 'val_len')

    def __init__(self, range_=FloatRange(-10.0, 10.0)):
        self._range = range_

    def __call__(self, *args, **kwargs) -> dict:
        result = {}
        for key in self.KEYS:
            length = self._adjust_or_get_len(key, **kwargs)
            result[key] = self._create_tensor(length)
        return result

    def _adjust_or_get_len(self, key: str, **kwargs) -> int:
        if key in kwargs and isinstance(kwargs.get(key), int):
            return kwargs.get(key) if self.MIN_LEN <= kwargs.get(key) <= self.MAX_LEN else self.DEFAULT_LEN
        return self.DEFAULT_LEN

    def _create_tensor(self, length: int) -> 'tensor':
        return torch.rand(length, 2) * (self._range.max - self._range.min) + self._range.min
