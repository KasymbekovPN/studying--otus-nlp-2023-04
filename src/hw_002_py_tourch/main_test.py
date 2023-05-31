import time

import torch
# import numpy
import matplotlib
# import matplotlib_inline
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


def loss(prediction, target):
    sq = (prediction - target) ** 2
    return sq.mean()


def predict(net, x, y):
    y_prediction = net.forward(x)

    plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    plt.plot(x.numpy(), y_prediction.data.numpy(), 'o', c='r', label='Prediction');
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()


class Net(torch.nn.Module):
    N0 = 1000
    N1 = 350
    N2 = 300
    N3 = 100

    def __init__(self):
        super(Net, self).__init__()
        self.fc0 = torch.nn.Linear(1, self.N0)
        # self.act0 = torch.nn.ReLU()
        self.act0 = torch.nn.Sigmoid()
        self.fc1 = torch.nn.Linear(self.N0, self.N1)
        # self.act1 = torch.nn.ReLU()
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(self.N1, self.N2)
        # self.act2 = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(self.N2, self.N3)
        # self.act3 = torch.nn.ReLU()
        self.act3 = torch.nn.Sigmoid()
        self.output = torch.nn.Linear(self.N3, 1)

    def forward(self, x):
        x = self.fc0(x)
        x = self.act0(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.output(x)
        return x


class SineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        # self.act1 = torch.nn.Sigmoid()
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

def demo0():
    matplotlib.rcParams['figure.figsize'] = (13.0, 5.0)

    # x_train = torch.rand(100)
    x_train = torch.rand(1000)
    x_train = x_train * 20 - 10


    y_train = torch.sin(x_train)

    # plt.plot(x_train.numpy(), y_train.numpy(), 'o')
    # plt.title('$y = sin(x)$')
    # plt.show()

    noise = torch.rand(y_train.shape) / 5
    # plt.plot(x_train.numpy(), noise.numpy(), 'o')

    # plt.axis([-10, 10, -1, 1])
    # plt.title('Gaussian noise');
    # plt.show()

    # y_train = y_train + noise

    # plt.plot(x_train.numpy(), y_train.numpy(), 'o')
    # plt.title('noisy sin(x)')
    # plt.xlabel('x_train')
    # plt.ylabel('y_train')
    # plt.show()

    x_train.unsqueeze_(1)
    y_train.unsqueeze_(1)
    # print(x_train)
    # print(y_train)

    x_validation = torch.linspace(-10, 10, 100)
    y_validation = torch.sin(x_validation.data)
    # plt.plot(x_validation.numpy(), y_validation.numpy(), 'o')
    # plt.title('sin(x)')
    # plt.xlabel('x_validation')
    # plt.ylabel('y_validation')
    # plt.show()

    x_validation.unsqueeze_(1)
    y_validation.unsqueeze_(1)

    # sine_net = SineNet(50)
    sine_net = Net()
    # predict(sine_net, x_validation, y_validation)

    optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.001)

    sine_net.train()
    for epoch in range(2000):
        if epoch % 100 == 0:
            print(f'epoch: {epoch}')
        optimizer.zero_grad()
        y_prediction = sine_net.forward(x_train)
        loss_val = loss(y_prediction, y_train)
        loss_val.backward()
        optimizer.step()

    sine_net.eval()
    predict(sine_net, x_validation, y_validation)


# for epoch_index in range(2000):
#     optimizer.zero_grad()
#
#     y_pred = sine_net.forward(x_train)
#     loss_val = loss(y_pred, y_train)
#
#     loss_val.backward()
#
#     optimizer.step()
#
# predict(sine_net, x_validation, y_validation)

def demo1():
    x = torch.rand(10)
    y = torch.rand(x.shape)
    print(x)
    print(y.numpy())
    #
    # z = x + y
    # print(z)
    #
    # a = torch.FloatTensor(2, 10)
    # print('a:', a)
    # for i in x:
    #     print('i: ', i)

    x.unsqueeze_(1)
    print(x)
    x.squeeze_(1)
    print(x)

    A_idx = torch.LongTensor([0]) # the index vector
    B = torch.LongTensor([[1, 2, 3], [4, 5, 6]])
    # B.squeeze_(1)
    C = B.index_select(1, A_idx)
    print(A_idx)
    print(B)
    print(C)
    C1 = C.detach().clone()
    C1.squeeze_(1)
    print(C)
    print(C1)


def demo2():
    b = np.arange(0.2, 3.2, 0.2)
    d = np.arange(0.1, 1.0, 0.1)
    nu = np.zeros((b.size, d.size))
    nu1 = np.zeros((b.size, d.size))
    counter_y = 0

    for deta in d:
        counter_x = 0
        for beta in b:
            nu[counter_x, counter_y] = math.sqrt(1 + (2 * deta * beta) ** 2) / math.sqrt(
                (1 - beta ** 2) ** 2 + (2 * deta * beta) ** 2)
            nu1[counter_x, counter_y] = 1 + math.sqrt(1 + (2 * deta * beta) ** 2) / math.sqrt(
                (1 - beta ** 2) ** 2 + (2 * deta * beta) ** 2)
            counter_x += 1
        counter_y += 1

    X, Y = np.meshgrid(d, b)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, nu)
    ax.plot_surface(X, Y, nu1)
    fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='b', marker='o')
    fake2Dline1 = matplotlib.lines.Line2D([1], [0], linestyle="none", c='orange', marker='o')
    ax.legend([fake2Dline, fake2Dline1], ['Lyapunov function on XY plane', '1234567'], numpoints = 1)
    plt.title('123')
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, nu)
    plt.title('12345')
    plt.show()


    # plt.plot(x.numpy(), y.numpy(), 'o', label='Groud truth')
    # plt.plot(x.numpy(), y_prediction.data.numpy(), 'o', c='r', label='Prediction');
    # plt.legend(loc='upper left')
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.show()

    # b = np.arange(0.2, 3.2, 0.2)
    # d = np.arange(0.1, 1.0, 0.1)
    #
    # B, D = np.meshgrid(b, d)
    # nu = np.sqrt( 1 + (2*D*B)**2 ) / np.sqrt( (1-B**2)**2 + (2*D*B)**2)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(B, D, nu)
    # plt.xlabel('b')
    # plt.ylabel('d')
    # plt.show()

def demo3():
    pass
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import random

    def fun(x, y):
      return 0.063*x**2 + 0.0628*x*y - 0.15015876*x + 96.1659*y**2 - 74.05284306*y  +      14.319143466051

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-1.0, 1.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fake2Dline = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker='o')
    ax.legend([fake2Dline], ['Lyapunov function on XY plane'], numpoints = 1)
    plt.show()


if __name__ == '__main__':
    pass
    # print('torch.HalfTensor', torch.HalfTensor)
    # print('torch.FloatTensor', torch.FloatTensor)
    # print('torch.DoubleTensor', torch.DoubleTensor)
    # print('torch.ShortTensor', torch.ShortTensor)
    # print('torch.IntTensor', torch.IntTensor)
    # print('torch.LongTensor', torch.LongTensor)
    # print('torch.CharTensor', torch.CharTensor)
    # print('torch.ByteTensor', torch.ByteTensor)

    # print('-' * 10)
    # print(torch.FloatTensor([1, 2]))
    # print('-' * 10)
    # ft1 = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    # print(ft1)
    # print(ft1.shape)
    # print('-' * 10)
    # ft2 = torch.FloatTensor(100)
    # print(ft2)
    # print('-' * 10)
    # ft3 = torch.FloatTensor(10, 10)
    # print(ft3)

    # ft4 = torch.FloatTensor(3, 2, 4)
    # print(ft4)
    # ft4.zero_()
    # print(ft4)
    # ft5 = torch.zeros(3, 2, 4)
    # print(ft5)
    # ft6 = torch.zeros_like(ft4)
    # print(ft6)
    # assert torch.allclose(ft4, ft5) and torch.allclose(ft4, ft6)

    # ft7 = torch.randn(2, 3)
    # print(ft7)
    #
    # print(ft7.random_(0, 10))                      # Дискретное равномерно U[0, 10]
    # print(ft7.uniform_(0, 1))                      # Равномерно U[0, 1]
    # print(ft7.normal_(mean=0, std=1))              # Нормальное со средним 0 и дисперсией 1
    # print(ft7.bernoulli_(p=0.5))                   # bernoulli with parameter p

    # ft8 = torch.FloatTensor(1024, 1024).uniform_()
    # print(ft8)
    # print(ft8.is_cuda)
    #
    # ft8.cuda()
    # time.sleep(5)
    # ft8.cpu()
    # torch.cuda.empty_cache()

    # demo0()
    # demo1()
    # demo2()
    # demo3()

    pass
    # t = torch.FloatTensor(2, 2)
    # print(t)
    # t[1, 1] = 123.9
    # print(t)
