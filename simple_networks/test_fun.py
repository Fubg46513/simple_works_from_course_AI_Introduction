"""
测试函数，注意地址修改，以及标签对应即可
"""

import torch
from torch import load, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


test_path = './test/' # 测试文件地址
input_size = [50,50]
batch_size = 32

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_fn = nn.CrossEntropyLoss()
# 可对model1~model3均作一次测试
# model = load("./models/model1.pth", weights_only=False)
# model = load("./models/model3.pth", weights_only=False)
model = load("./models/model2.pth", weights_only=False)
model.eval()
test_losses = []
test_accuracies = []

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    test_losses.append(test_loss)
    test_accuracies.append(correct * 100)
    print(f"Test:\nAccuracy: {(100 * correct):>0.1f}%,loss: {test_loss:>8f} \n")




test_transforms = transforms.Compose([
    transforms.Resize(input_size),  # 将输入图片resize成统一尺寸
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
    transforms.ToTensor(),          # 将PIL Image或numpy.ndarray转换为tensor，并归一化到[0,1]之间
])


dataset = ImageFolder(test_path,transform=test_transforms)
testdata_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)


test(testdata_loader, model, loss_fn)