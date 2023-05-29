import torch

from src.hw_002_py_tourch.datasource.float_range import FloatRange


class ComplexLinearDS:
    MIN_LEN = 1
    MAX_LEN = 1_000_000
    DEFAULT_LEN = 10_000

    def __init__(self, range_=FloatRange(10.0, -10.0)):
        self._range = range_

    def __call__(self, *args, **kwargs) -> tuple:
        train_len = self._adjust_or_get_len('train_len', **kwargs)
        test_len = self._adjust_or_get_len('test_len', **kwargs)
        val_len = self._adjust_or_get_len('val_len', **kwargs)

        # return (torch.rand(train_len) * self._range.min)
        # train_len: int,
        # test_len: int,
        # val_len: int,
        pass

    def _adjust_or_get_len(self, key: str, **kwargs) -> int:
        if key in kwargs and isinstance(kwargs.get(key), int):
            return kwargs.get(key) if self.MIN_LEN <= kwargs.get(key) <= self.MAX_LEN else self.DEFAULT_LEN
        return self.DEFAULT_LEN

    def _adjust_range(self, tensor: 'tensor') -> 'tensor':
        pass
