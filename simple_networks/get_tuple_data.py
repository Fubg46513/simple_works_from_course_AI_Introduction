"""
代码主要为了测试hook函数的效果以及使用hook提取某一输入其某一层的输入数据与输出数据方便对网络进行可视化处理
"""


import torch
from PIL import Image
from torch import load
from torchvision import transforms

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并移动到指定设备
model = load("./models/model2.pth", weights_only=False)
print(model)
model = model.to(device)

# 打开输入图像
input_image = Image.open(r'.\Animal Images\cats\_4d47ddc.jpg')
input_size = [50, 50]

# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize(input_size),  # 将输入图片resize成统一尺寸
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
])

# 预处理输入图像
input_tensor = preprocess(input_image)

# 添加批量维度
input_tensor = input_tensor.unsqueeze(0)

# 将输入张量移动到指定设备
input_tensor = input_tensor.to(device)

# 用于存储钩子函数的输入和输出
features_in_hook = []
features_out_hook = []

# 定义钩子函数
def hook(model, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    print(fea_out.shape)

# 注册钩子函数
layer_name = 'fc2'
for (name, module) in model.named_modules():
    if name == layer_name:
        module.register_forward_hook(hook=hook)

# 禁止梯度计算
with torch.no_grad():
    # 进行模型推理
    output = model(input_tensor)

def save_tuple_to_txt(tup, file_path):
    try:
        # 将元组元素转换为字符串并用空格连接
        tuple_str = ' '.join(map(str, tup))
        with open(file_path, 'w') as file:
            file.write(tuple_str)
        print(f"元组已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存文件时出现错误: {e}")

# 保存文件的路径
file_path = 'tuple_data.txt'
save_tuple_to_txt(features_in_hook[0], file_path)



# # 打印钩子函数的输出形状
# if features_out_hook:
#     print(features_out_hook[0].shape)
#     print(features_in_hook)
#     print(features_out_hook)
