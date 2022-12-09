import torchvision.transforms as T


def pil_transforms(image_size):
    return T.Compose([
        T.Resize(size=(image_size, image_size)),
        T.ToTensor()
    ])
