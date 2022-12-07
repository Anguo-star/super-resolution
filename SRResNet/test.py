import os

import torch

import configs
from models.srresnet import SRResNet
from utils.transform_util import pil_transforms
from utils.dataset_util import gen_dataset
from utils.draw_image import plot_images

start_epoch = 0
def test_super_resolution(num_plots: int = 1, checkpoint_path: str = configs.CHECKPOINT_PATH, device=configs.DEVICE):
    model = SRResNet(in_channels=configs.INPUT_CHANNELS,
                     feature_map_channels=configs.FEATURE_MAP_CHANNELS,
                     num_residual_layers=configs.NUM_RESIDUAL_LAYERS,
                     scale_factor=configs.UPSCALE_FACTOR)
    if os.path.exists(checkpoint_path):
        global start_epoch
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        print(f"epoch: {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1

    dataset = gen_dataset(
        root=configs.LFW_IMAGE_PATH,
        hr_transforms=pil_transforms(image_size=configs.IMAGE_SIZE),
        lr_transforms=pil_transforms(image_size=configs.IMAGE_SIZE // configs.UPSCALE_FACTOR))

    for i in range(num_plots):
        hr_image, lr_image = dataset[i]
        sr_image = model(lr_image.unsqueeze(0)).squeeze(0)

        plot_images(lr_image.permute((1, 2, 0)).cpu().detach().numpy(),
                    hr_image.permute((1, 2, 0)).cpu().detach().numpy(),
                    sr_image.permute((1, 2, 0)).cpu().detach().numpy())


if __name__ == '__main__':
    test_super_resolution(2, r'results/saved_model/SRResNet_256_64x4_16res.pth')
    print(f'start_epoch: {start_epoch}')


