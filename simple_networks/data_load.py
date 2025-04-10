"""
训练数据与验证数据集的导入，引入了随机种子确保可复现。
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split

# 设置随机数种子以确保可复现性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# 一些基本参数
data_path = './Animal Images/'
input_size = [50,50]
rate = 0.7 #训练集占整个数据集大小


train_transforms = transforms.Compose([
    transforms.Resize(input_size),  # 将输入图片resize成统一尺寸
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
    transforms.ToTensor(),          # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
])

full_dataset = datasets.ImageFolder(data_path,transform=train_transforms)

train_size = int(rate * len(full_dataset))
test_size = len(full_dataset) - train_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# 创建数据加载器
batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")