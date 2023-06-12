import math


def sin_exp_function(x, y):
    return math.sin(x + 2 * y) * math.exp(-1 * ((2 * x + y) ** 2))
