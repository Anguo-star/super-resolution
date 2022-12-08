import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs


def get_optimizer(model, optimizer_name: str = configs.OPTIMIZER, learning_rate: float = configs.LEARNING_RATE):
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4)


def get_lr_scheduler(optimizer, schedule: str = configs.LR_SCHEDULE):
    try:
        if schedule.lower() == 'StepLR'.lower():
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=configs.STEP_SIZE, gamma=configs.LR_GAMA)
        elif schedule.lower() == 'MultistepLR'.lower():
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=configs.MILESTONES, gamma=configs.LR_GAMA)
        elif schedule.lower() == 'ExponentialLR'.lower():
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=configs.LR_GAMA)
        elif schedule.lower() == 'CosineAnnealingLR'.lower():
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=configs.COS_T_MAX, eta_min=configs.COS_ETA_MIN)
        elif schedule.lower() == 'LambdaLR'.lower():
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=configs.LR_LAMBDA, )
    except:
        raise ValueError(f"""schedule {schedule} not support, please choose one from:
                         ['StepLR', 'MultistepLR', 'ExponentialLR', 'CosineAnnealingLR']""")
