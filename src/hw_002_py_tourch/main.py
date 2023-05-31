import torch
import matplotlib
import matplotlib.pyplot as plt

from src.hw_002_py_tourch.datasource.linear_ds import ComplexLinearDS
from src.hw_002_py_tourch.function.functions import sin_exp_function


def create_figure(args: 'tensor', results: list, title: str) -> 'plt':

    # todo move into function
    idx0 = torch.LongTensor([0])
    idx1 = torch.LongTensor([1])

    buffer = args.detach().clone()
    x = buffer.index_select(1, idx0).squeeze_(1)
    y = buffer.index_select(1, idx1).squeeze_(1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # x_ = x.numpy()
    # y_ = y.numpy()
    for result in results:
        # r_ = result.numpy()
        # ax.plot_surface(x_, y_, r_)
        ax.plot_surface(x.numpy(), y.numpy(), result.numpy())
    # todo ???
    # fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    # fake2Dline1 = matplotlib.lines.Line2D([1], [0], linestyle="none", c='orange', marker='o')
    # ax.legend([fake2Dline, fake2Dline1], ['Lyapunov function on XY plane', '1234567'], numpoints = 1)
    plt.title(title)


if __name__ == '__main__':
    train_len = 140
    test_len = 30
    val_len = 30

    ds_result = ComplexLinearDS()(train=train_len, test=test_len, val=val_len)
    train_args = ds_result.get('train')
    test_args = ds_result.get('test')
    val_args = ds_result.get('val')
    # print(len(train_args))
    # print(len(test_args))
    # print(len(val_args))

    train_func_result = sin_exp_function(train_args)
    # print(len(func_result))
    # print(func_result)

    create_figure(train_args, [train_func_result], 'some title').show()


    pass
