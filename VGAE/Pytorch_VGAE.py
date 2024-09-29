import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, VGAE
from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from 

class GINEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GINEncoder, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        ))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)

class VGAEModel(VGAE):
    def __init__(self, in_channels, out_channels):
        encoder = GINEncoder(in_channels, out_channels)
        super(VGAEModel, self).__init__(encoder)

    def encode(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.reparametrize(z, z)

root = './VGAE_dataset'
dataset = CustomGraphDataset(root=root)
# 处理数据（如果尚未处理）
if not os.path.exists(dataset.processed_paths[0]):
    dataset.process()

# 获取数据集的第一个图数据对象以确定 in_channels
data = dataset[0]
in_channels = data.x.shape[1]
out_channels = 16  # 你可以根据需要调整这个值

transform = RandomLinkSplit(is_undirected=True)
train_data_list = []
val_data_list = []
test_data_list = []