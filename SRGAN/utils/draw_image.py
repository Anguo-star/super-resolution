import cv2
import matplotlib.pyplot as plt


def plot_images(lr_image, hr_image, sr_image):

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(lr_image)

    plt.subplot(132)
    plt.imshow(hr_image)

    plt.subplot(133)
    plt.imshow(sr_image)
    plt.show()


if __name__ == '__main__':
    import os
    import sys
    from dataset_util import gen_dataset
    from transform_util import pil_transforms
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import configs
    from models.srresnet import SRResNet

    root = os.path.realpath(os.path.join('../', configs.LFW_IMAGE_PATH))
    dataset = gen_dataset(root,
                          hr_transforms=pil_transforms(configs.IMAGE_SIZE),
                          lr_transforms=pil_transforms(
                          configs.IMAGE_SIZE // configs.UPSCALE_FACTOR))
    hr_image, lr_image = dataset[0]
    model = SRResNet(in_channels=configs.INPUT_CHANNELS,
                     feature_map_channels=configs.FEATURE_MAP_CHANNELS,
                     num_residual_layers=configs.NUM_RESIDUAL_LAYERS,
                     scale_factor=configs.UPSCALE_FACTOR)
    sr_image = model(lr_image.unsqueeze(0)).squeeze(0)

    plot_images(lr_image.permute((1, 2, 0)).cpu().detach().numpy(),
                hr_image.permute((1, 2, 0)).cpu().detach().numpy(),
                sr_image.permute((1, 2, 0)).cpu().detach().numpy())
