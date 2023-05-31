import torch


def slice_two_args_tensor(tensor: 'tensor') -> tuple:
    idx0 = torch.LongTensor([0])
    idx1 = torch.LongTensor([1])

    buffer = tensor.detach().clone()
    x = buffer.index_select(1, idx0).squeeze_(1)
    y = buffer.index_select(1, idx1).squeeze_(1)

    return x, y