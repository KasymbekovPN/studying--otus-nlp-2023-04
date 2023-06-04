import torch
import matplotlib
import matplotlib.pyplot as plt

from src.hw_002_py_tourch.configurator.configurator import Configurator
from src.hw_002_py_tourch.data.points import Points
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


def train(neuron_net, points: 'Points', configurator: 'Configurator'):
    optimizer = torch.optim.Adam(neuron_net.parameters(), lr=configurator('learning-rate'))
    neuron_net.train()

    epoch_quantity = configurator('quantity.epochs')
    for idx, epoch in enumerate(range(epoch_quantity)):
        if epoch % configurator('step.epoch-log') == 0:
            print(f'[TRAIN] {idx * 100.0 / epoch_quantity}% {idx} / {epoch_quantity}')
        optimizer.zero_grad()
        z_predication = neuron_net.forward(points.torch_input)
        loss_val = compute_mse_loss(z_predication, points.torch_output)
        loss_val.backward()
        optimizer.step()
    neuron_net.eval()
    print(f'[TRAIN] Done')


def test():
    pass


def display(x, y, z_calculated, z_approximated):
    calculated_data = {'value': z_calculated, 'color': 'blue', 'text': 'Calculated'}
    approximated_data = {'value': z_approximated, 'color': 'orangered', 'text': 'Approximated'}

    create_figures(x, y, [calculated_data], title='Calculated')
    create_figures(x, y, [approximated_data], title='Approximated')
    create_figures(x, y, [calculated_data, approximated_data], 'Comparison').show()


if __name__ == '__main__':
    conf = Configurator()
    net = Net(conf('quantity.neurons'))
    train_points = Points(conf('size.train.x'), conf('size.train.y'))

    train(net, train_points, conf)

    result_ravel = net.forward(train_points.torch_input).squeeze(1).detach().numpy()
    result_mesh = result_ravel.reshape(train_points.mesh_x.shape)
    display(train_points.mesh_x, train_points.mesh_y, train_points.mesh_z, result_mesh)
