
def compute_mse_loss(prediction, target):
    return ((prediction - target) ** 2).mean()
