import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, VGAE
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.optim import Adam
import time
import scipy.sparse as sp

class GINEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINEncoder, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        ))
        self.conv_mu = GINConv(nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        ))
        self.conv_logstd = GINConv(nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        ))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd

class VGAEModel(VGAE):
    def __init__(self, in_channels, out_channels):
        encoder = GINEncoder(in_channels, out_channels)
        super(VGAEModel, self).__init__(encoder)

    def encode(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        self.__mu__ = mu
        self.__logstd__ = logstd
        z = self.reparametrize(mu, logstd)
        return z

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        
        for i in range(5000):  # 假设有 5000 个图
            # 读取邻接矩阵 A
            A_path = os.path.join(self.raw_dir, 'A', f'A_{i}.csv')
            A = pd.read_csv(A_path, header=None).values
            A = sp.coo_matrix(A)
            edge_index, _ = from_scipy_sparse_matrix(A)

            # 读取特征矩阵 X
            X_path = os.path.join(self.raw_dir, 'X', f'X_{i}.csv')
            X = pd.read_csv(X_path, header=None).values
            X = torch.tensor(X, dtype=torch.float)

            # 创建图数据对象
            data = Data(x=X, edge_index=edge_index)
            data_list.append(data)

        # 将所有图对象保存到单个文件中
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 数据集实例化
root = './VGAE_dataset'
dataset = CustomGraphDataset(root=root)

# 超参数
num_epochs = 200
learning_rate = 0.01
batch_size = 32
out_channels = 16  # 嵌入维度

# 创建 DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 实例化模型和优化器
model = VGAEModel(in_channels=dataset[0].x.size(1), out_channels=out_channels)
optimizer = Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    start_time = time.time()  # 记录开始时间
    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 从图数据中获取节点特征和边索引
        x, edge_index = data.x, data.edge_index
        
        # 前向传播
        z = model.encode(x, edge_index)
        
        # 重建损失
        loss = model.recon_loss(z, edge_index)
        kl_divergence = model.kl_loss()
        total_loss = loss + kl_divergence
        
        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 打印每个批次的损失
        #print(f'  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {total_loss.item():.4f}')

        total_loss += total_loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time  # 计算经过时间
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s')

# 保存训练后的模型
torch.save(model.state_dict(), 'vgae_model.pth')

# 确认使用了所有的数据
print(f'Total training samples used: {len(dataset)}')
