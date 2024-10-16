import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        # 第一个全连接层，将输入维度从 10 转换为 64
        self.fc1 = nn.Linear(10, 64)
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(64)
        # 第二个全连接层，将维度从 64 转换为 32
        self.fc2 = nn.Linear(64, 32)
        # 输出层，将维度从 32 转换为 1
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        # 输入通过第一个全连接层
        x = self.fc1(x)
        # 通过批归一化
        x = self.bn1(x)
        # ReLU 激活
        x = nn.ReLU()(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        # 输出层
        x = self.output(x)
        return x