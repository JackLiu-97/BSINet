import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import transforms as T

res_shape = (256, 256)


class SegmentationPresetTrain:
    def __init__(self, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=256):
        trans = []
        trans.append(T.RandomRotation(0.5))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, image1, image2, target):
        return self.transforms(image1, image2, target)


def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return SegmentationPresetTrain(mean=mean, std=std)


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.img_path_A = glob.glob(os.path.join(data_path, 'A', '*.png'))
        self.img_path_B = glob.glob(os.path.join(data_path, 'B', '*.png'))
        self.mask_path = glob.glob(os.path.join(data_path, 'label', '*.png'))
        self.transforms = get_transform()

    def __getitem__(self, index):
        images1 = Image.open(self.img_path_A[index])
        images2 = Image.open(self.img_path_B[index])
        labels = np.array(Image.open(self.mask_path[index])) / 255
        labels = Image.fromarray(labels)
        if self.transforms is not None:
            images1, images2, labels = self.transforms(images1, images2, labels)
        return images1, images2, labels

    def __len__(self):
        return len(self.img_path_A)


def Mydataset_collate(batch):
    images1 = []
    images2 = []
    masks = []
    for images1, images2, masks in batch:
        images1.append(images1)
        images2.append(images2)
        masks.append(masks)
    images1 = np.array(images1)
    images2 = np.array(images2)
    masks = np.array(masks)
    return images1, images2, masks
