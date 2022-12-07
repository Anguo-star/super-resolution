import random
import sys
import os

from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from PIL import Image


class LFWDataset(Dataset):
    def __init__(self, paths, hr_transforms, lr_transforms):
        self.paths = paths
        self.hr_transforms = hr_transforms
        self.lr_transforms = lr_transforms
        if not (self.hr_transforms or self.lr_transforms):
            raise ValueError("lack of hr_transforms or lr_transforms!")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        image = Image.open(img_path, mode='r').convert('RGB')
        dataset = []
        if self.hr_transforms:
            hr_image = self.hr_transforms(image)
            dataset.append(hr_image)
        if self.lr_transforms:
            lr_image = self.lr_transforms(image)
            dataset.append(lr_image)
        return dataset


def load_remote_dataset(dataset='celeba'):
    return torchvision.datasets.CelebA(root='../data/CelebA', split='train', download=True)


def gen_dataset(root, hr_transforms, lr_transforms, split=None):
    """
    生成数据集
    :param split:
    :param root: 文件跟目录
    :param transforms: transform函数
    :param target_transforms: target_transforms函数
    :return: Dataset数据集
    """
    img_paths = get_subpaths(root)
    dataset = LFWDataset(img_paths, hr_transforms, lr_transforms)
    if split and isinstance(split, float):
        train_dataset, test_dataset = split_dataset(dataset, split)
        return train_dataset, test_dataset
    return dataset


def get_subpaths(root: str):
    """
    根据root路径，输出该路径下的全部子文件路径列表
    :param root: 输入路径
    :return: 输出子文件列表 subpaths
    """
    paths = []
    for subroots, subdirs, subfiles in os.walk(os.path.realpath(root)):
        for file in subfiles:
           paths.append(os.path.join(subroots, file))
    return paths


def split_indexes(length, split: float = 0.8):
    """
    Determine the specified segmentation index proportion of training and testing data set.
    :param length: length of the whole input dataset.
    :param split: split percentage for train dataset, and the rest for test.
    :return: splited indexes of train and test.
    """
    split_index = int(length * split)
    indexes = list(range(length))
    random.shuffle(indexes)
    train_indexes = indexes[: split_index]
    test_indexes = indexes[split_index:]
    return train_indexes, test_indexes


def split_dataset(dataset, split: float = 0.8):
    """
    split dataset into to part: train_dataset and test_dataset
    :param dataset: dataset prepared to be split
    :param split: split ratio that indicate the percentage of train dateset and the rest is test dataset
    :return: splited train_dataset and test_dataset
    """
    train_indexes, test_indexes = split_indexes(len(dataset), split)
    train_dataset = Subset(dataset, train_indexes)
    test_dataset = Subset(dataset, test_indexes)
    return train_dataset, test_dataset


def transfer_files(source_root, to_root):
    from shutil import copy
    paths = get_subpaths(source_root)
    for path in paths:
        copy(path, to_root)


if __name__ == '__main__':
    from transform_util import pil_transforms

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import configs

    root = os.path.realpath(os.path.join('../', configs.LFW_IMAGE_PATH))
    train_dataset, test_dataset = gen_dataset(root,
                                             hr_transforms=pil_transforms(configs.IMAGE_SIZE),
                                             lr_transforms=pil_transforms(configs.IMAGE_SIZE // configs.UPSCALE_FACTOR),
                                             split=0.8)
    print(train_dataset[0][0].shape, test_dataset[0][1].shape)
    print(len(train_dataset), len(test_dataset))
    dataloader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
    print(next(iter(dataloader))[0].shape)


    # transfer_files(root, r'D:\资料\学习资料\References\Super-resolution\Face SR\dataset\lfw_flatten')
