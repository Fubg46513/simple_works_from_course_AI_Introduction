"""
共使用了三个神经网络进行了训练测试，分别为Model1--简单神经网络，Model2--示例神经网络，Model3--resnet18
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class Model1(nn.Module):
    def __init__(self, input_shape = None, num_classes = 2):
        super(Model1, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        # 第一个卷积层
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(self._calculate_fc_input_shape(), 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _calculate_fc_input_shape(self):
        x = torch.randn(1, *self.input_shape)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        # 卷积层前向传播
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层前向传播
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x



class Model2(nn.Module):
    def __init__(self, input_shape = None):
        super(Model2, self).__init__()
        if input_shape is None:
            input_shape = [1, 50, 50]
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._calculate_fc_input_size(input_shape), 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def _calculate_fc_input_size(self, input_shape):
        x = torch.randn(1, *input_shape)
        x = self._conv_block(x, self.conv1, self.relu1, self.pool1)
        x = self._conv_block(x, self.conv2, self.relu2, self.pool2)
        x = self.dropout1(x)
        x = self._conv_block(x, self.conv3, self.relu3, None)
        x = self.dropout2(x)
        x = self.flatten(x)
        return x.size(1)

    @staticmethod
    def _conv_block(x, conv, relu, pool=None):
        x = conv(x)
        x = relu(x)
        if pool is not None:
            x = pool(x)
        return x

    def forward(self, x):
        x = self._conv_block(x, self.conv1, self.relu1, self.pool1)
        x = self._conv_block(x, self.conv2, self.relu2, self.pool2)
        x = self.dropout1(x)
        x = self._conv_block(x, self.conv3, self.relu3, None)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x


class Model3(nn.Module):
    def __init__(self, input_shape=None, num_classes=2):
        super(Model3, self).__init__()
        # 导入预训练的 ResNet18 模型
        if input_shape is None:
            input_shape = [1, 50, 50]
        self.resnet18 = models.resnet18(weights = ResNet18_Weights.DEFAULT)
        # 获取输入通道数
        in_channels = input_shape[0]
        # 修改第一层卷积层以适应输入通道数
        if in_channels != 3:
            self.resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后一层全连接层以适应自定义的类别数
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


class Model2v2(nn.Module):
    def __init__(self, input_shape = None):
        super(Model2v2, self).__init__()
        if input_shape is None:
            input_shape = [1, 50, 50]
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.7)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.7)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._calculate_fc_input_size(input_shape), 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def _calculate_fc_input_size(self, input_shape):
        x = torch.randn(1, *input_shape)
        x = self._conv_block(x, self.conv1, self.relu1, self.pool1)
        x = self._conv_block(x, self.conv2, self.relu2, self.pool2)
        x = self.dropout1(x)
        x = self._conv_block(x, self.conv3, self.relu3, None)
        x = self.dropout2(x)
        x = self.flatten(x)
        return x.size(1)

    @staticmethod
    def _conv_block(x, conv, relu, pool=None):
        x = conv(x)
        x = relu(x)
        if pool is not None:
            x = pool(x)
        return x

    def forward(self, x):
        x = self._conv_block(x, self.conv1, self.relu1, self.pool1)
        x = self._conv_block(x, self.conv2, self.relu2, self.pool2)
        x = self.dropout1(x)
        x = self._conv_block(x, self.conv3, self.relu3, None)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x