import math

import torch
from torch import nn
import torchvision


class Discriminator(nn.Module):
    """判别器"""
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, num_blocks=4, adaptive_pool_size=6, dense_size=1024):
        super(Discriminator, self).__init__()

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.leaky_relu_input = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(num_blocks):
            if i == 0:
                blocks.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False))
                blocks.append(nn.BatchNorm2d(out_channels))
            else:
                blocks.append(
                    nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride=1, padding=1, bias=False))
                out_channels *= 2
                blocks.append(nn.BatchNorm2d(out_channels))

        self.conv_blocks = nn.Sequential(*blocks)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(adaptive_pool_size, adaptive_pool_size))

        self.dense_1 = nn.Linear(out_channels * adaptive_pool_size * adaptive_pool_size, dense_size)
        self.leaky_relu_output = nn.LeakyReLU(0.2, inplace=True)
        self.dense_2 = nn.Linear(dense_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                num = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / num))
                if module.bias:
                    module.bias.data.zero_()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = self.leaky_relu_input(self.conv_input(inputs))
        output = self.conv_blocks(output)
        output = self.adaptive_pool(output)
        output = self.leaky_relu_output(self.dense_1(output.view(batch_size, -1)))
        output = self.sigmoid(self.dense_2(output))
        return output


class TruncatedVGG19(nn.Module):
    def __init__(self, i, j):
        """
        param i: 第i个池化层
        param j: 第j个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        self.vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_loc = 1
        for layer in self.vgg19.features.children():
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0
            if maxpool_counter == i - 1 and conv_counter == j:
                break
            truncate_loc += 1

        self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children()))[:truncate_loc]

    def forward(self, inputs):
        output = self.truncated_vgg19(inputs)
        return output


if __name__ == '__main__':
    vgg19 = torchvision.models.vgg19()
    print(list(vgg19.features.children()))
    print(list(vgg19.features.modules()))
