import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import configs
from models.srresnet import SRResNet
from utils.optimizer_util import get_optimizer, get_lr_scheduler
from utils.loss_util import get_loss_func
from utils.transform_util import pil_transforms
from utils.dataset_util import gen_dataset


logging.getLogger().setLevel(logging.INFO)


# 全局变量
start_epoch = 0
writer = SummaryWriter()


def main():
    cudnn.benchmark = True  # 对卷积进行加速

    random.seed(configs.RANDOM_SEED)
    torch.manual_seed(configs.RANDOM_SEED)

    train_dataset, test_dataset = gen_dataset(
        root=configs.LFW_IMAGE_PATH,
        hr_transforms=pil_transforms(image_size=configs.IMAGE_SIZE),
        lr_transforms=pil_transforms(image_size=configs.IMAGE_SIZE // configs.UPSCALE_FACTOR),
        split=0.8)
    logging.info(f"length of train_dataset is {len(train_dataset)}, length of test_dataset is {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=configs.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=configs.NUM_WORKERS)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=configs.NUM_WORKERS)

    model = get_model(configs.GENERATOR)

    history = train(model, train_dataloader, test_dataloader, configs)

    return history


def train(model, train_dataloader, test_dataloader, configs):
    global writer
    logging.info("[INFO] Training start...")

    num_epochs = configs.NUM_EPOCHS
    optimizer = get_optimizer(model)
    lr_scheduler = get_lr_scheduler(optimizer)
    loss_func = get_loss_func()

    num_batch = len(train_dataloader)
    history = {}
    lr_history = []
    train_losses = []
    valid_losses = []

    if configs.OPEN_CHECKPOINT_READ:
        load_checkpoint(model, optimizer, lr_scheduler)
    for epoch in range(start_epoch, num_epochs):
        for batch, (hr_imgs, lr_imgs) in enumerate(train_dataloader):
            hr_imgs, lr_imgs = hr_imgs.to(configs.DEVICE), lr_imgs.to(configs.DEVICE)
            sr_imgs, train_loss = train_step(model, lr_imgs, hr_imgs, optimizer, loss_func)
            train_losses.append(train_loss)

            valid_loss = validation(model, test_dataloader, loss_func)
            valid_losses.append(valid_loss)


            if batch == 0:
                writer.add_image(
                    f"epoch_{str(epoch)}_lr", make_grid(lr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image(
                    f"epoch_{str(epoch)}_sr", make_grid(sr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image(
                    f"epoch_{str(epoch)}_hr", make_grid(hr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
            if batch % 1 == 0 or batch == num_batch - 1:
                print(f"[{epoch + 1}/{num_epochs}] [{batch + 1}/{num_batch}]\nTrain Loss: {train_loss}\tValid Loss: {valid_loss}\tlr: {lr_scheduler.state_dict()['_last_lr']}")

        del lr_imgs, hr_imgs, sr_imgs
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Valid_loss', valid_loss, epoch)

        lr_history.append(lr_scheduler.state_dict()['_last_lr'])
        lr_scheduler.step()


        if configs.OPEN_CHECKPOINT_WRITE:
            logging.info(f"[INFO] Saving checkpoint model (epoch={epoch+1}) start...")
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict()},
                       configs.MODEL_PATH)
            logging.info(f"[INFO] Saving checkpoint model (epoch={epoch+1}) over!")

    logging.info("[INFO] Train over!")
    logging.info("[INFO] Saving model start...")
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()},
               configs.MODEL_PATH)
    logging.info("[INFO] Saving model over!")

    history['lr_history'] = lr_history
    history['train_losses'] = train_losses
    history['valid_losses'] = valid_losses
    np.save(configs.HISTORY_PATH, history)

    writer.close()
    return history


def train_step(model, x_data, y_data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_data)
    loss = loss_fn(y_pred, y_data)
    loss.backward()
    optimizer.step()
    return y_pred, loss


def validation(model, test_dataloader, loss_fn):
    model.eval()
    num_batch = len(test_dataloader)
    with torch.no_grad():
        losses = 0
        for y_data, x_data in test_dataloader:
            y_data, x_data = y_data.to(configs.DEVICE), x_data.to(configs.DEVICE)
            y_pred = model(x_data)
            loss = loss_fn(y_data, y_pred)
            losses += loss
        del x_data, y_data, y_pred
    return losses / num_batch


def get_model(generator=configs.GENERATOR,
              device=configs.DEVICE,
              ngpu=configs.NGPU):
    logging.info(f"selected model is {configs.GENERATOR}")
    if generator.lower() == 'srresnet':
        model = SRResNet(in_channels=configs.INPUT_CHANNELS,
                         feature_map_channels=configs.FEATURE_MAP_CHANNELS,
                         num_residual_layers=configs.NUM_RESIDUAL_LAYERS,
                         scale_factor=configs.UPSCALE_FACTOR)
    else:
        model = SRResNet(in_channels=configs.INPUT_CHANNELS,
                         feature_map_channels=configs.FEATURE_MAP_CHANNELS,
                         num_residual_layers=configs.NUM_RESIDUAL_LAYERS,
                         scale_factor=configs.UPSCALE_FACTOR)
    model = model.to(device)
    if ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))
    return model


def load_checkpoint(model, optimizer, lr_scheduler):
    global start_epoch
    if os.path.exists(configs.CHECKPOINT_PATH):
        checkpoint = torch.load(configs.CHECKPOINT_PATH, map_location=configs.DEVICE)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1


if __name__ == '__main__':
    history = main()
    print(history)

