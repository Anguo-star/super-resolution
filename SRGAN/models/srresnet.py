import math

import torch
from torch import nn


class SRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, in_channels, feature_map_channels, num_residual_layers, scale_factor):
        super(SRResNet, self).__init__()

        # 输入层卷积
        self.conv_input = nn.Conv2d(in_channels, feature_map_channels, kernel_size=9, stride=1, padding=4, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        # 残差块
        self.residual_block = self.make_layer(ResidualLayer(feature_map_channels, kernel_size=3), num_residual_layers)

        # 中间层卷积块
        self.conv_mid = nn.Conv2d(feature_map_channels, feature_map_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance_norm_mid = nn.InstanceNorm2d(feature_map_channels, affine=True)

        # pixel_shuffle放大
        self.upscale_layers = []
        while scale_factor > 1:
            self.upscale_layers.append(UpscaleLayer(feature_map_channels))
            scale_factor //= 2
        self.upscale_block = nn.Sequential(*self.upscale_layers)

        # 输出卷积块
        self.conv_ioutput = nn.Conv2d(feature_map_channels, in_channels, kernel_size=9, stride=1, padding=4, bias=False)
        self.tanh = nn.Tanh()

        # 初始化参数
        self.init_weights()

    def make_layer(self, block, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block)
        return nn.Sequential(*layers)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                num = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / num))
                if module.bias:
                    module.bias.data.zero_()

    def forward(self, inputs):
        output = self.leaky_relu(self.conv_input(inputs))
        output = self.residual_block(output)
        output = self.instance_norm_mid(self.conv_mid(output))
        output = self.upscale_block(output)
        output = self.tanh((self.conv_ioutput(output)))
        return output


class ResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size, **kwargs):
        super(ResidualLayer, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1, bias=False)
        self.instance_norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=1, padding=1, bias=False)
        self.instance_norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, inputs):
        output1 = self.leaky_relu(self.instance_norm1(self.conv1(inputs)))
        output2 = self.instance_norm2(self.conv2(output1))
        return torch.add(output2, inputs)


class UpscaleLayer(nn.Module):
    """
    尺寸放大为原来的2倍
    """
    def __init__(self, in_channels):
        super(UpscaleLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.upscale = nn.PixelShuffle(2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inputs):
        output = self.conv(inputs)
        output = self.upscale(output)
        output = self.relu(output)
        return output


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import configs

    model = SRResNet(in_channels=configs.INPUT_CHANNELS,
                     feature_map_channels=configs.FEATURE_MAP_CHANNELS,
                     num_residual_layers=configs.NUM_RESIDUAL_LAYERS,
                     scale_factor=configs.UPSCALE_FACTOR)
    print(model)
    print(model.state_dict())
    # inputs = torch.randn(size=(16, 3, 250, 250))
    # output = model(inputs)
    # print(output.shape)
