import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import configs


def get_optimizer(model, optimizer_name: str = configs.OPTIMIZER, learning_rate: float = configs.LEARNING_RATE):
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=5e-4)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=5e-4)


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
        raise ValueError(f"""schedule {schedule} not support, please choose one from followings:
                         ['StepLR', 'MultistepLR', 'ExponentialLR', 'CosineAnnealingLR', 'LambdaLR']""")


if __name__ == '__main__':
    optimizer = get_optimizer(model=torch.nn.Conv2d(3, 3, 3, 1, 1))
    lr_scheduler = get_lr_scheduler(optimizer)
    print(optimizer.state_dict())
    print(lr_scheduler.state_dict()['_last_lr'])
