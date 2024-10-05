import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,VGAE
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
import scipy.sparse as sp
import numpy as np



def adj_to_edge_index(adj):
    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)
    adj_coo = adj.tocoo()
    row = torch.tensor(adj_coo.row, dtype=torch.long)
    col = torch.tensor(adj_coo.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

# 准备 PyG 数据集并分割训练/测试边
def prepare_data(X, A):
    data_list = []
    for i in range(len(X)):
        x = torch.tensor(X[i], dtype=torch.float)
        edge_index = adj_to_edge_index(A[i])
        data = Data(x=x, edge_index=edge_index)
        data = train_test_split_edges(data)
        data_list.append(data)
    return data_list

class GINEdgeAggregator(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GINEdgeAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_weight=None):
        #x = add_self_loops(edge_index, num_nodes=x.size(0))[0]  # 添加自环
        out = GINConv(self.mlp, train_eps=True)(x, edge_index, edge_weight)
        return out

class VGAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(VGAEEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # 第一个 GIN 层
        self.convs.append(GINEdgeAggregator(in_channels, hidden_channels))

        # 后续的 GIN 层
        for _ in range(1, num_layers):
            self.convs.append(GINEdgeAggregator(hidden_channels, hidden_channels))

        # 输出层
        self.fc_mu = nn.Linear(hidden_channels, out_channels)
        self.fc_log_sigma = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        mu = self.fc_mu(x)
        log_sigma = self.fc_log_sigma(x)
        
        # 通过重参数化技巧进行采样
        z = self.reparameterize(mu, log_sigma)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std


