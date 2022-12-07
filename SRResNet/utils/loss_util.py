import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs


def get_loss_func(loss_name: str = configs.LOSS_FN, device=configs.DEVICE):
    if loss_name == 'MSE':
        return nn.MSELoss().to(device)


if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), '..'))
    import configs

    y_data = torch.randn(size=(2, 3, 250, 250))
    y_pred = torch.randn(size=(2, 3, 250, 250))
    loss = get_loss_func(configs.LOSS_FN)(y_data, y_pred)
    loss_2 = torch.sqrt(torch.square(y_data - y_pred)).mean(1).mean(1).mean()
    loss_3 = F.mse_loss(y_pred, y_data)
    print(loss, loss_2, loss_3)
