import os
import random
import warnings
import numpy as np
import torch
from models.mynet.BSINet import MyNet
from PIL import Image
from sklearn.metrics import confusion_matrix
import transforms as T

random.seed(47)


class DataPrese:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = []
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, image1, image2, target):
        return self.transforms(image1, image2, target)


def get_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return DataPrese(mean=mean, std=std)


warnings.filterwarnings("ignore")

"""
SNUnet r"D:\模型权重文件\建筑物\20230412-033255\SNUnet.pth"
"D:\模型权重文件\建筑物\20230412-072830\FC_EF.pth"

GZ:"D:\模型权重文件\建筑物\GZ\20230531-105508\best.pth" 88.4
D:\Datasets\Data_CD\CD_Data_GZ\CD_Data_GZ\256\test 
D:\Download\SiamUnet_diff_best (3).pth 89.2
"D:\Download\SiamUnet_diff_best (4).pth" val 87.22 最好

WHU:"D:\模型权重文件\建筑物\WHU\20230525-205745\best.pth"
D:\Datasets\Data_CD\WHU\1\out\test


"""


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--ckpt_url", default=r"D:\Download\SiamUnet_diff_best (6).pth",
                        help="data root")
    parser.add_argument("--modelname", default="",
                        help="data root")
    parser.add_argument("--data_path", default=r"D:\Datasets\Data_CD\WHU\1\out\3333\test",
                        help="data root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--out_path", default=r"C:\LangChao\b_detection\test\result\mynet", help="val root")
    args = parser.parse_args()

    return args


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
model = MyNet(3, 2)

weights_dict = torch.load(args.ckpt_url, map_location=torch.device('cuda'))
model.load_state_dict(weights_dict['model'])
model.eval()
model.to(device)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

transform = get_transform()
# 预测并保存结果
testA_dir = os.path.join(args.data_path, "A")
testB_dir = os.path.join(args.data_path, "B")
label_dir = os.path.join(args.data_path, "label")
result_dir = args.out_path
numbers = len(os.listdir(testA_dir))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
for filename in os.listdir(testA_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 加载图像
        image_path_A = os.path.join(testA_dir, filename)
        image_path_B = os.path.join(testB_dir, filename)
        label_path = os.path.join(label_dir, filename)
        # label = np.array(Image.open(label_path))
        label = (np.array(Image.open(label_path)) / 255).astype("uint8")
        label = Image.fromarray(label)
        imageA = Image.open(image_path_A)
        imageB = Image.open(image_path_B)
        imageA, imageB, label = transform(imageA, imageB, label)
        imageA, imageB = imageA.to(device), imageB.to(device)
        # 推理
        with torch.no_grad():
            output = model(imageA.unsqueeze(0), imageB.unsqueeze(0))
            output = torch.argmax(output, dim=1).squeeze(0)
            output = output.detach().cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(label.flatten(), output.flatten(), labels=[0, 1]).ravel()
        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp
tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
