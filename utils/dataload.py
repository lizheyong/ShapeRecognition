import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader

transforms = transforms.Compose([
    transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
])

path = r'.\dataset\circle'
data_train = datasets.ImageFolder(path, transform=transforms)

data_loader = DataLoader(data_train, batch_size=64, shuffle=True)

for i, data in enumerate(data_loader):
    images, labels = data

    print(images.shape)
    print(labels.shape)
    break
