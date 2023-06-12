import numpy
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.hw_002_py_tourch.configurator.configurator import Configurator
from src.hw_002_py_tourch.data.points import Points
from src.hw_002_py_tourch.data.args import create_random_float_args
from src.hw_002_py_tourch.loss.functions import compute_mse_loss
from src.hw_002_py_tourch.nn.neuro_net import Net


def create_figures(x, y, z_data: list, title: str) -> 'plt':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lines = []
    texts = []
    for idx, z_datum in enumerate(z_data):
        ax.plot_surface(x, y, z_datum['value'], color=z_datum['color'])
        lines.append(matplotlib.lines.Line2D([idx], [0], linestyle="none", c=z_datum['color'], marker='o'))
        texts.append(z_datum['text'])
    ax.legend(lines, texts, numpoints=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    return plt


def train(neuron_net, points: Points, configurator: 'Configurator'):
    optimizer = torch.optim.Adam(neuron_net.parameters(), lr=configurator('learning-rate'))
    neuron_net.train()

    epoch_quantity = configurator('quantity.epochs')
    for idx, epoch in enumerate(range(epoch_quantity)):
        if epoch % configurator('step.epoch-log') == 0:
            print(f'\r[TRAIN] {idx * 100.0 / epoch_quantity}% {idx} / {epoch_quantity}', end='')
        optimizer.zero_grad()
        z_predication = neuron_net.forward(points.torch_input)
        loss_val = compute_mse_loss(z_predication, points.torch_output)
        loss_val.backward()
        optimizer.step()
    neuron_net.eval()
    print(f'\r[TRAIN] Done')


def test(neuron_net, points: Points, tag: str, threshold=0.0):
    approximated_ravel = neuron_net.forward(points.torch_input).squeeze(1).detach().numpy()
    calculated_ravel = points.ravel_z

    if threshold > 0.0:
        multiplier = (approximated_ravel + calculated_ravel) / 2.0
        multiplier[multiplier <= threshold] = 0.0
        multiplier[multiplier > threshold] = 1.0
        approximated_ravel = approximated_ravel * multiplier
        calculated_ravel = calculated_ravel * multiplier

    mse = compute_mse_loss(approximated_ravel, calculated_ravel)
    print(f'[TEST::{tag}] MSE = {mse}')


def display(neuron_net, points: Points, tag: str) -> plt:
    result_ravel = neuron_net.forward(points.torch_input).squeeze(1).detach().numpy()
    result_mesh = result_ravel.reshape(points.mesh_x.shape)

    calculated_data = {'value': points.mesh_z, 'color': 'blue', 'text': 'Calculated'}
    approximated_data = {'value': result_mesh, 'color': 'orangered', 'text': 'Approximated'}

    x = points.mesh_x
    y = points.mesh_y
    create_figures(x, y, [calculated_data], title=f'[{tag}] Calculated')
    create_figures(x, y, [approximated_data], title=f'[{tag}] Approximated')
    create_figures(x, y, [calculated_data, approximated_data], f'[{tag}] Comparison')

    return plt


if __name__ == '__main__':
    conf = Configurator()
    net = Net(conf('quantity.neurons'))

    train_points = Points(
        create_random_float_args(conf('size.train.x')),
        create_random_float_args(conf('size.train.y'))
    )
    train(net, train_points, conf)

    threshold = 0.0
    test_points_r = Points(
        create_random_float_args(conf('size.test.x')),
        create_random_float_args(conf('size.test.y'))
    )
    test(net, test_points_r, f'RANDOM threshold={threshold}')

    test_points_l = Points(
        numpy.linspace(-10.0, 10.0, conf('size.test.x')),
        numpy.linspace(-10.0, 10.0, conf('size.test.y'))
    )
    test(net, test_points_l, f'LINEAR threshold={threshold}')

    val_points_r = Points(
        create_random_float_args(conf('size.val.x')),
        create_random_float_args(conf('size.val.y'))
    )
    display(net, val_points_r, 'RANDOM')

    val_points_l = Points(
        numpy.linspace(-10.0, 10.0, conf('size.val.x')),
        numpy.linspace(-10.0, 10.0, conf('size.val.y'))
    )
    display(net, val_points_l, 'LINEAR').show()
