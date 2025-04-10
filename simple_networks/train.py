"""
训练函数，相关参数可以直接进行修改，同时三个模型也可以相应更改尝试不同模型训练。
"""

import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from data_load import train_loader, test_loader
from model import Model1,Model2,Model3,Model2v2

# 相关参数
epochs = 50
batch_size = 32
channel = 1

width = 50
height = 50
input_shape = (channel, width, height)

# 检查是否有可用的 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化模型并将其移到 GPU 上
model = Model2v2(input_shape).to(device)
loss_fun = nn.CrossEntropyLoss()
opt = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)

# 用于记录训练和测试过程中的损失和准确率
train_losses = []
test_losses = []
test_accuracies = []


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % 100 == 0:
            loss_val, current = loss.item(), batch * len(X)
            print(f"Train: loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")
    # 计算该轮训练的平均损失
    avg_loss = running_loss / len(dataloader)
    train_losses.append(avg_loss)


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


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_loader, model, loss_fun, opt)
    test(test_loader, model, loss_fun)
print("Done!")

# 绘制训练和测试损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

torch.save(model, './models/model2v2_drop0.7.pth')