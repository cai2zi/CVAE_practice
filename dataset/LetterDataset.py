import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor


class LetterDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.transforms = transforms

    def __getitem__(self, index):
        c='A';d={}
        for i in range(26):
            tmp=chr(ord(c)+i)
            d[tmp]=i
        img_path = self.imgs[index]
        label = os.path.basename(img_path)[0]  # 获取字母标签
        data = Image.open(img_path).convert('L')  # 转换为灰度图
        if self.transforms:
            data = self.transforms(data)
        return data, d[label]

    def __len__(self):
        return len(self.imgs)