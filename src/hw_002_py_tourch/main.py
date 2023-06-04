import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from src.hw_002_py_tourch.configurator.configurator import Configurator
from src.hw_002_py_tourch.data.linear_ds import ComplexLinearDS
from src.hw_002_py_tourch.data.args import create_float_args
from src.hw_002_py_tourch.data.array_processor import group_line_float_tensors
from src.hw_002_py_tourch.data.points import Points
from src.hw_002_py_tourch.function.functions import sin_exp_function
from src.hw_002_py_tourch.loss.functions import compute_mse_loss
from src.hw_002_py_tourch.nn.neuro_net import Net



def create_figure_old(x: 'ndarray', y: 'ndarray', zz: list, title: str) -> 'plt':

    # # todo move into function
    # idx0 = torch.LongTensor([0])
    # idx1 = torch.LongTensor([1])
    #
    # buffer = args.detach().clone()
    # x = buffer.index_select(1, idx0).squeeze_(1)
    # y = buffer.index_select(1, idx1).squeeze_(1)

    def fun(x_, y_):
        return 0.063*x_**2 + 0.0628*x_*y_ - 0.15015876*x_ + 96.1659*y_**2 - 74.05284306*y_ + 14.319143466051

    def func1(x_, y_):
        return np.sin(x + 2 * y) * np.exp(-1 * ((2 * x + y) ** 2))

    def func2(x_, y_):
        return math.sin(x_ + 2 * y_) * math.exp(-1 * ((2 * x_ + y_) ** 2))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # x = y = np.arange(-1.0, 1.0, 0.05)
    # x = y = np.arange(-10.0, 10.0, 0.05)

    x = np.arange(-10.0, 10.0, 0.05)
    y = np.arange(-1.0, 1.0, 0.05)

    X, Y = np.meshgrid(x, y)
    # zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    # zs = np.array([func1(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    zs = np.array([func2(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fake2Dline = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker='o')
    ax.legend([fake2Dline], ['Lyapunov function on XY plane'], numpoints = 1)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # x_ = x.numpy()
    # # y_ = y.numpy()
    # for z in zz:
    #     # r_ = result.numpy()
    #     # ax.plot_surface(x_, y_, r_)
    #     ax.plot_surface(x, y, z)
    # # todo ???
    # # fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    # # fake2Dline1 = matplotlib.lines.Line2D([1], [0], linestyle="none", c='orange', marker='o')
    # # ax.legend([fake2Dline, fake2Dline1], ['Lyapunov function on XY plane', '1234567'], numpoints = 1)
    # plt.title(title)
    return plt


def create_figure(x, y, z, title: str) -> 'plt':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    return plt


def train(neuron_net, points: 'Points', configurator: 'Configurator'):
    optimizer = torch.optim.Adam(neuron_net.parameters(), lr=configurator('learning-rate'))
    neuron_net.train()

    for epoch in range(configurator('quantity.epochs')):
        if epoch % configurator('step.epoch-log') == 0:
            print(f'epoch: {epoch}')
        optimizer.zero_grad()
        z_predication = neuron_net.forward(points.torch_input)
        loss_val = compute_mse_loss(z_predication, points.torch_output)
        loss_val.backward()
        optimizer.step()
    neuron_net.eval()


if __name__ == '__main__':
    conf = Configurator()
    net = Net(conf('quantity.neurons'))
    train_points = Points(conf('size.train.x'), conf('size.train.y'))
    train(net, train_points, conf)

    result_ravel = net.forward(train_points.torch_input).squeeze(1).detach().numpy()
    result_mesh = result_ravel.reshape(train_points.mesh_x.shape)

    create_figure(train_points.mesh_x, train_points.mesh_y, train_points.mesh_z, 'Calculated')
    create_figure(train_points.mesh_x, train_points.mesh_y, result_mesh, 'Approximated').show()
