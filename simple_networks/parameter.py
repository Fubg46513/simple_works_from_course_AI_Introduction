from torchsummary import summary
import torch

from model import Model2

# 实例化模型
input_shape = (1,50,50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model2(input_shape).to(device)

# 打印模型参数信息
summary(model, input_size=(1,50,50))