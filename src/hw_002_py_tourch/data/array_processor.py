import enum

import torch


class Message(enum.Enum):
    EMPTY_ARGS = 'Args is empty'
    DISALLOWED_ARG = 'Some arg(s) have disallowed'
    BAD_DIM = 'Tensor has bad dimension'
    EMPTY_TENSORS = 'Tensors are empty'
    DIFF_LENS = 'Tensors have difference lengths'


def group_line_float_tensors(*args) -> 'tensor':
    message = None
    args_len = len(args)
    if args_len == 0:
        message = Message.EMPTY_ARGS
    vector_len = -1
    for arg in args:
        if type(arg) != torch.Tensor:
            message = Message.DISALLOWED_ARG
            break
        if len(arg.shape) != 1:
            message = Message.BAD_DIM
            break
        if len(arg) == 0:
            message = Message.EMPTY_TENSORS
            break
        current_len = len(arg)
        if vector_len == -1:
            vector_len = current_len
        else:
            if vector_len != current_len:
                message = Message.DIFF_LENS
                break

    if message is not None:
        raise Exception(message.value)

    result = torch.FloatTensor(vector_len, args_len)
    for i in range(0, vector_len):
        for j in range(0, args_len):
            result[i][j] = args[j][i]
    return result
