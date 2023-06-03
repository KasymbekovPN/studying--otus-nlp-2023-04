import unittest
import torch
import random

from parameterized import parameterized
from src.common.test import repeat
from src.hw_002_py_tourch.nn.neuro_net import Net, create_layer, create_activator


class TestCase(unittest.TestCase):

    @parameterized.expand([
        (Net.MIN_NEURON_QUANTITY - 1, Net.DEFAULT_NEURON_QUANTITY),
        (Net.MIN_NEURON_QUANTITY, Net.MIN_NEURON_QUANTITY),
        (Net.MIN_NEURON_QUANTITY + 1, Net.MIN_NEURON_QUANTITY + 1),
        (Net.MAX_NEURON_QUANTITY - 1, Net.MAX_NEURON_QUANTITY - 1),
        (Net.MAX_NEURON_QUANTITY, Net.MAX_NEURON_QUANTITY),
        (Net.MAX_NEURON_QUANTITY + 1, Net.DEFAULT_NEURON_QUANTITY),
    ])
    def test_check_or_get_quantity(self, quantity: int, expected: int):
        net = Net()
        result = net._checkOrGetQuantity(quantity)
        self.assertEqual(expected, result)

    @repeat(1_000)
    def test_layer_creation(self):
        in_features = random.randint(1, 10)
        out_features = random.randint(1, 10)
        layer = create_layer(in_features, out_features)

        self.assertEqual(torch.nn.Linear, type(layer))
        self.assertEqual(in_features, layer.in_features)
        self.assertEqual(out_features, layer.out_features)

    def test_activator_creation(self):
        activator = create_activator()
        self.assertEqual(torch.nn.ReLU, type(activator))


if __name__ == '__main__':
    unittest.main()
