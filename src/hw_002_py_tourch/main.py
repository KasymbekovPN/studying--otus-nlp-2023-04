import time

import torch
# import numpy
import matplotlib
# import matplotlib_inline
import matplotlib.pyplot as plt


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
    N0 = 100
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

    x_train = torch.rand(100)
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

    y_train = y_train + noise
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

    optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)

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

    demo0()
