import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GINConv
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops

class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINEncoder, self).__init__()
        # GINConv requires an MLP for its message passing mechanism
        self.mlp_shared = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(0.1),  # 增加额外的层
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.mlp_mu = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.gin_shared = GINConv(self.mlp_shared)
        self.gin_mu = GINConv(self.mlp_mu)
        self.gin_logvar = GINConv(self.mlp_logvar)

    def forward(self, x, edge_index):
        x = F.relu(self.gin_shared(x, edge_index))
        mu = self.gin_mu(x, edge_index)
        logvar = self.gin_logvar(x, edge_index)
        return mu, logvar


class DeepVGAE(VGAE):
    def __init__(self, in_channels:int,hidden_channels:int,out_channels:int):
        super(DeepVGAE, self).__init__(encoder=GINEncoder(in_channels=in_channels,
                                                          hidden_channels=hidden_channels,
                                                          out_channels=out_channels),
                                       decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = torch.sigmoid(self.decoder.forward_all(z))  # 使用 sigmoid
        return adj_pred

    
    def loss(self, x, pos_edge_index):
        z = self.encode(x, pos_edge_index)
        
        # 计算正边的损失
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()
        # 直接进行负样本采样
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0), num_neg_samples=pos_edge_index.size(1) * 3)
        # 计算负边的损失
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()
        # 计算KL散度损失
        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss 


    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score
