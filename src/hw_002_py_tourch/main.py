import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from src.hw_002_py_tourch.datasource.linear_ds import ComplexLinearDS
from src.hw_002_py_tourch.datasource.args import create_float_args


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


# todo move
def sin_exp_func(x, y):
    return math.sin(x + 2 * y) * math.exp(-1 * ((2 * x + y) ** 2))


# todo ndarray
def create_figure(x: 'ndarray', y: 'ndarray', z: 'ndarray', title: str) -> 'plt':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z_mesh)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)
    # fake2Dline = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker='o')
    # ax.legend([fake2Dline], ['Lyapunov function on XY plane'], numpoints = 1)

    return plt


if __name__ == '__main__':
    train_len = 140
    test_len = 30
    val_len = 30

    x_original = create_float_args(train_len)
    y_original = create_float_args(train_len)

    x_mesh, y_mesh = np.meshgrid(x_original, y_original)
    x_ravel = np.ravel(x_mesh)
    y_ravel = np.ravel(y_mesh)
    z_ravel = np.array([sin_exp_func(x, y) for x, y in zip(x_ravel, y_ravel)])
    z_mesh = z_ravel.reshape(x_mesh.shape)
    create_figure(x_mesh, y_mesh, z_mesh, 'Train').show()


    # ds_result = ComplexLinearDS()(train=train_len, test=test_len, val=val_len)
    # train_args = ds_result.get('train')
    # test_args = ds_result.get('test')
    # val_args = ds_result.get('val')
    # print(len(train_args))
    # print(len(test_args))
    # print(len(val_args))

    # train_func_result = sin_exp_function(train_args)
    # print(len(func_result))
    # print(func_result)

    # create_figure(train_args, [train_func_result], 'some title').show()


    pass
