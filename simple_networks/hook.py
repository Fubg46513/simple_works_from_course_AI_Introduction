"""
在hook文件夹下的所有图片导入模型中，寻找最后一个64->2线性层中输入张量（1*64）与目标的张量相似度最高（欧氏距离最小）的前几个图片。
主要目的还是看看最后一层分类前模型聚类的效果，防止模型最后仍然猫狗不分。
"""

import torch
from torch import load
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import shutil
from scipy.spatial.distance import euclidean

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并移动到指定设备
model = load("./models/model2.pth", weights_only=False)
model = model.to(device)

# 定义预处理转换
input_size = [50, 50]
preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# 创建临时分类目录
temp_class_dir = './hook_temp_class'
if not os.path.exists(temp_class_dir):
    os.makedirs(temp_class_dir)
    class_dir = os.path.join(temp_class_dir, 'class_0')
    os.makedirs(class_dir)
    # 复制图片到临时分类目录
    for img_file in os.listdir('./hook'):
        if img_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            src = os.path.join('./hook', img_file)
            dst = os.path.join(class_dir, img_file)
            shutil.copyfile(src, dst)

# 加载数据集
train_dataset = ImageFolder(root=temp_class_dir, transform=preprocess)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# 目标张量,为了方便起见直接使用笨办法对数据导入，做一个简单的验证
target_tensor = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0232, 0.0000, 0.0000, 0.2082,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.1446, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0118, 0.0000, 0.0806, 0.0000, 0.0000, 0.0000, 0.0018, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.1016, 0.0208, 0.0000, 0.0000, 0.0045, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0559, 0.0000, 0.0295, 0.1341,
                               0.6382, 0.0000, 0.0000, 0.1352, 0.3387, 0.0000, 0.0000, 0.0000, 0.0500,
                               0.0000, 0.3067, 0.0000, 0.0820, 0.0000, 0.0000, 0.0231, 0.7256, 0.0000,
                               0.0000]], device=device)
target_np = target_tensor.cpu().numpy().flatten()

# 用于存储钩子函数的输入和对应的图像路径
features_in_hook = []
image_paths = []


# 定义钩子函数
def hook(model, fea_in, fea_out):
    input_tensor = fea_in[0].cpu().detach().squeeze(0)  # 转换到 CPU 并去除批量维度
    if input_tensor.shape == (64,):
        features_in_hook.append(input_tensor)


# 注册钩子函数
layer_name = 'fc2'
for (name, module) in model.named_modules():
    if name == layer_name:
        module.register_forward_hook(hook=hook)

# 禁止梯度计算
with torch.no_grad():
    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        img_path = train_dataset.samples[i][0]
        image_paths.append(img_path)
        _ = model(data)

# 计算与目标张量的欧氏距离
distances = []
for tensor in features_in_hook:
    tensor_np = tensor.numpy()
    dist = euclidean(target_np, tensor_np)
    distances.append(dist)

# 找出距离最小的 4 个张量的索引
indices = sorted(range(len(distances)), key=lambda i: distances[i])[:5]

# 获取对应的图像地址
nearest_image_paths = [image_paths[i] for i in indices]

print("与目标张量欧氏距离最小的四个张量所对应的图片文件地址：")
for path in nearest_image_paths:
    print(path)

# 删除临时目录
shutil.rmtree(temp_class_dir)
