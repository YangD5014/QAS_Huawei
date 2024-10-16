import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class DeepBinaryAutoencoder(nn.Module):
    def __init__(self, input_dim=(42, 22), latent_dim=10):
        super(DeepBinaryAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim[0] * input_dim[1], 512),  # 增加隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 防止过拟合
            nn.Linear(512, 256),  # 新增隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout
            nn.Linear(256, latent_dim)  # 连接到潜在层
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout
            nn.Linear(256, 512),  # 新增隐藏层
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout
            nn.Linear(512, input_dim[0] * input_dim[1]),
            nn.Sigmoid(),  # 输出压缩到 [0, 1]
            nn.Unflatten(1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 模型、损失函数和优化器
model = DeepBinaryAutoencoder()
criterion = nn.BCELoss()  # 二值矩阵使用 Binary Cross Entropy 作为损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 加载数据（假设 input_data 是 shape 为 (batch_size, 42, 22) 的二值矩阵）
X_list = []
for i in range(6000):
    X = pd.read_csv(f'../VGAE_dataset/raw/X/X_{i}.csv', header=None).values
    X_list.append(X)

A_np = np.array(X_list)
input_data = torch.tensor(A_np).float()

# 训练步骤
for epoch in range(5000):  # 训练500个epoch
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, input_data)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 保存模型参数
model_path = "deep_binary_autoencoder.pth"
torch.save(model.state_dict(), model_path)
print(f'Model parameters saved to {model_path}')
