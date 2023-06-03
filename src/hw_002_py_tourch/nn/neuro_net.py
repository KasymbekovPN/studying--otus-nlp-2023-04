import torch


def create_layer(in_features: int, out_features: int):
    return torch.nn.Linear(in_features, out_features)


def create_activator():
    return torch.nn.ReLU()


class Net(torch.nn.Module):
    MIN_NEURON_QUANTITY = 1
    MAX_NEURON_QUANTITY = 1_000_000
    DEFAULT_NEURON_QUANTITY = 100
    NET_IN_FEATURES = 2
    NET_OUT_FEATURES = 1
    DEFAULT_NEURON_QUANTITIES = [DEFAULT_NEURON_QUANTITY, DEFAULT_NEURON_QUANTITY, DEFAULT_NEURON_QUANTITY]

    def __init__(self,
                 neuron_quantities=None,
                 layer_creator=create_layer,
                 activator_creator=create_activator) -> None:
        super(Net, self).__init__()
        if neuron_quantities is None or len(neuron_quantities) == 0:
            neuron_quantities = self.DEFAULT_NEURON_QUANTITIES
        in_features = self.NET_IN_FEATURES
        self._layers = []
        for q in neuron_quantities:
            self._layers.append(layer_creator(in_features, q))
            self._layers.append(activator_creator())
            in_features = q
        self._layers.append(create_layer(in_features, self.NET_OUT_FEATURES))

    # todo check x type
    def forward(self, x):
        for layer in self._layers:
            print(x)
            x = layer(x)
        return x

    def _checkOrGetQuantity(self, original_value: int) -> int:
        return original_value \
            if self.MIN_NEURON_QUANTITY <= original_value <= self.MAX_NEURON_QUANTITY \
            else self.DEFAULT_NEURON_QUANTITY
