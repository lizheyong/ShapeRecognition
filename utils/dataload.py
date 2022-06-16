import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.ImageFolder(root='../dataset', transform=train_transform)
    loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)

    print(dataset.class_to_idx)

    # 所有图片的路径和对应的label
    print(dataset.imgs)

    # 没有任何的transform，所以返回的还是PIL Image对象
    # print(dataset[0][1])# 第一维是第几张图，第二维为1返回label
    # print(dataset[0][0]) # 为0返回图片数据
    print(dataset[299][1])
