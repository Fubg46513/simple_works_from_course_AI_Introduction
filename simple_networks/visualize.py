"""
主要做的是对model2的三次卷积操作可视化，观察卷积后图像的结果
"""
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch import load
from torchvision import transforms


input_image = Image.open(r'.\Animal Images\cats\_4d47ddc.jpg')
input_size = [50, 50]

preprocess = transforms.Compose([
    transforms.Resize(input_size),  # 将输入图片resize成统一尺寸
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
    transforms.ToTensor(),  # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

model = load("./models/model2.pth", weights_only=False)
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')
with torch.no_grad():
    output = model(input_batch)

# 存储每次卷积层的输出
conv_outputs = []
x = input_batch
conv_count = 0
for name, layer in model.named_children():
    if isinstance(layer, torch.nn.Conv2d):
        x = layer(x)
        conv_outputs.append(x.cpu())
        conv_count += 1
        if conv_count == 3:
            break
    else:
        x = layer(x)

# 分别对三次卷积结果进行可视化
for i, conv_output in enumerate(conv_outputs):
    num_features = conv_output.shape[1]
    print(f"total of number of feature maps in conv {i + 1}: ", num_features)
    plt.figure(figsize=(20, 17))
    for j in range(min(num_features, 32)):
        plt.subplot(4, 8, j + 1)
        plt.axis('off')
        plt.imshow(conv_output[0, j, :, :].detach(), cmap='gray')
    #plt.title(f'Feature maps of Conv {i + 1}')
    plt.show()


