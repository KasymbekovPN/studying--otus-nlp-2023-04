import torch


def sin_exp_function(tensor: 'tensor') -> 'tensor':
    idx0 = torch.LongTensor([0])
    idx1 = torch.LongTensor([1])

    buffer = tensor.detach().clone()
    x = buffer.index_select(1, idx0).squeeze_(1)
    y = buffer.index_select(1, idx1).squeeze_(1)

    return torch.sin(x + 2 * y) * torch.exp(-1 * ((2 * x + y) ** 2))
