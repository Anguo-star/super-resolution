import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs


def get_optimizer(model, optimizer_name: str = configs.OPTIMIZER, learning_rate: float = configs.LEARNING_RATE):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
