import numpy
import torch

from src.hw_002_py_tourch.function.functions import sin_exp_function
from src.hw_002_py_tourch.data.array_processor import group_line_float_tensors
from src.hw_002_py_tourch.data.args import create_float_args


class Points:
    def __init__(self, x_size: int, y_size: int) -> None:
        self._original_x = create_float_args(x_size)
        self._original_y = create_float_args(y_size)
        self._mesh_x, self._mesh_y = numpy.meshgrid(self._original_x, self._original_y)
        self._ravel_x = numpy.ravel(self._mesh_x)
        self._ravel_y = numpy.ravel(self._mesh_y)
        self._ravel_z = numpy.array([sin_exp_function(x, y) for x, y in zip(self._ravel_x, self._ravel_y)])
        self._mesh_z = self._ravel_z.reshape(self._mesh_x.shape)
        self._torch_input = group_line_float_tensors(torch.from_numpy(self._ravel_x), torch.from_numpy(self._ravel_y))
        self._torch_output = torch.from_numpy(self._ravel_z).unsqueeze_(1)

    @property
    def original_x(self):
        return self._original_x

    @original_x.setter
    def original_x(self, value):
        raise Exception('Original X setting unsupported')

    @property
    def original_y(self):
        return self._original_y

    @original_y.setter
    def original_y(self, value):
        raise Exception('Original Y setting unsupported')

    @property
    def mesh_x(self):
        return self._mesh_x

    @mesh_x.setter
    def mesh_x(self, value):
        raise Exception('Mesh X setting unsupported')

    @property
    def mesh_y(self):
        return self._mesh_y

    @mesh_y.setter
    def mesh_y(self, value):
        raise Exception('Mesh Y setting unsupported')

    @property
    def mesh_z(self):
        return self._mesh_z

    @mesh_z.setter
    def mesh_z(self, value):
        raise Exception('Mesh Z setting unsupported')

    @property
    def ravel_x(self):
        return self._ravel_x

    @ravel_x.setter
    def ravel_x(self, value):
        raise Exception('Ravel X setting unsupported')

    @property
    def ravel_y(self):
        return self._ravel_y

    @ravel_y.setter
    def ravel_y(self, value):
        raise Exception('Ravel Y setting unsupported')

    @property
    def ravel_z(self):
        return self._ravel_z

    @ravel_z.setter
    def ravel_z(self, value):
        raise Exception('Ravel Z setting unsupported')

    @property
    def torch_input(self):
        return self._torch_input

    @torch_input.setter
    def torch_input(self, value):
        raise Exception('TORCH INPUT setting unsupported')

    @property
    def torch_output(self):
        return self._torch_output

    @torch_output.setter
    def torch_output(self, value):
        raise Exception('TORCH OUTPUT setting unsupported')
