import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np


class FER2013ForXception(Dataset):
    def __init__(self, root='../data', mode='train', transform=None):
        self.root = root
        # 定义标准转换：转为 Tensor 并归一化到 [0, 1]
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # 常用标准化，让像素值在 [-1, 1]
        ])

        assert mode in ['train', 'val', 'test']
        self.mode = mode

        # 读取 CSV
        csv_path = os.path.join(self.root, 'fer2013.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到数据集文件: {csv_path}")

        self.df = pd.read_csv(csv_path)

        # 划分数据集
        if self.mode == 'train':
            self.df = self.df[self.df['Usage'] == 'Training']
        elif self.mode == 'val':
            self.df = self.df[self.df['Usage'] == 'PrivateTest']
        else:
            self.df = self.df[self.df['Usage'] == 'PublicTest']

    def __getitem__(self, index: int):
        data_series = self.df.iloc[index]
        label = data_series['emotion']
        pixels = data_series['pixels']

        # 将字符串像素转换为 numpy 数组 (48x48)
        face = np.fromstring(pixels, sep=' ', dtype=np.uint8).reshape(48, 48)

        if self.transform:
            # transforms.ToTensor() 会自动处理 HWC/HW 到 CHW 的转换并缩放到 [0,1]
            face = self.transform(face)

        return face, label

    def __len__(self) -> int:
        return len(self.df)



def get_dataloaders(root_path, batch_size=64):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = FER2013ForXception(root_path, mode='train', transform=train_transform)
    val_set = FER2013ForXception(root_path, mode='val', transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader