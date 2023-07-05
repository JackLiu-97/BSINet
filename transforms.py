import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image1, image2, target):
        for t in self.transforms:
            image1, image2, target = t(image1, image2, target)
        return image1, image2, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image1, image2, target):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image1 = F.resize(image1, size)
        image2 = F.resize(image2, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image1, image2, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.flip_prob:
            image1 = F.hflip(image1)
            image2 = F.hflip(image2)
            target = F.hflip(target)
        return image1, image2, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.flip_prob:
            image1 = F.vflip(image1)
            image2 = F.vflip(image2)
            target = F.vflip(target)
        return image1, image2, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image1, image2, target):
        image1 = pad_if_smaller(image1, self.size)
        image2 = pad_if_smaller(image2, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image1, (self.size, self.size))
        image1 = F.crop(image1, *crop_params)

        crop_params = T.RandomCrop.get_params(image2, (self.size, self.size))
        image2 = F.crop(image1, *crop_params)

        target = F.crop(target, *crop_params)
        return image1, image2, target


# class CenterCrop(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, image1, image2, target):
#         image1 = F.center_crop(image1, self.size)
#         image2 = F.center_crop(image2, self.size)
#         target = F.center_crop(target, self.size)
#         return image1, image2, target


class ToTensor(object):
    def __call__(self, image1, image2, target):
        image1 = F.to_tensor(image1)
        image2 = F.to_tensor(image2)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image1, image2, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image1, image2, target):
        image1 = F.normalize(image1, mean=self.mean, std=self.std)
        image2 = F.normalize(image2, mean=self.mean, std=self.std)
        return image1, image2, target


class GaussianBlur(object):
    def __init__(self, Blur_prob):
        self.Blur_prob = Blur_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.Blur_prob:
            GaussianBlur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
            image1 = GaussianBlur(image1)
            image2 = GaussianBlur(image2)

        return image1, image2, target


class GrayScale(object):
    def __init__(self, gray_prob):
        self.gray_prob = gray_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.gray_prob:
            Grayscale = T.Grayscale(num_output_channels=3)
            image1 = Grayscale(image1)
            image2 = Grayscale(image2)
        return image1, image2, target


class ColorJitter(object):
    def __init__(self, color_prob):
        self.color_prob = color_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.color_prob:
            ColorJitter = T.ColorJitter(brightness=.2, contrast=.2, hue=.2)
            image1 = ColorJitter(image1)
            image2 = ColorJitter(image2)
        return image1, image2, target


class RandomAdjustSharpness(object):
    def __init__(self, sharp_prob):
        self.sharp_prob = sharp_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.sharp_prob:
            RandomAdjustSharpness = T.RandomAdjustSharpness(sharpness_factor=0, p=1)
            image1 = RandomAdjustSharpness(image1)
            image2 = RandomAdjustSharpness(image2)
        return image1, image2, target


class RandomEqualize(object):
    def __init__(self, equalize_prob):
        self.equalize_prob = equalize_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.equalize_prob:
            RandomEqualize = T.RandomEqualize(p=1)
            image1 = RandomEqualize(image1)
            image2 = RandomEqualize(image2)
        return image1, image2, target


class RandomRotation(object):
    def __init__(self, rotation_prob):
        self.rotation_prob = rotation_prob

    def __call__(self, image1, image2, target):
        if random.random() < self.rotation_prob:
            if random.random() < 0.5:
                RandomRotation = T.RandomRotation(degrees=(45, 45), expand=False)
                image1 = RandomRotation(image1)
                image2 = RandomRotation(image2)
                target = RandomRotation(target)
            else:
                RandomRotation = T.RandomRotation(degrees=(30, 30), expand=False)
                image1 = RandomRotation(image1)
                image2 = RandomRotation(image2)
                target = RandomRotation(target)
        return image1, image2, target


class CenterCrop(object):
    def __init__(self, crop_prob, size):
        self.crop_prob = crop_prob
        self.size = size

    def __call__(self, image1, image2, target):
        if random.random() < self.crop_prob:
            CenterCrop = T.CenterCrop(self.size)
            image1 = CenterCrop(image1)
            image2 = CenterCrop(image2)
            target = CenterCrop(target)
        return image1, image2, target
