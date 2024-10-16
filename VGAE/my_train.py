import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import pandas as pd
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data, DataLoader
from GVAE_GIN import DeepVGAE
import json

# 将邻接矩阵转换为 PyTorch Geometric 支持的 edge_index 格式
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

# 模型训练函数
def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    # 修改loss调用，去掉all_edge_index
    loss = model.loss(data.x, data.train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

# 模型测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)

        # 添加检查
        if data.test_pos_edge_index.numel() == 0 or data.test_neg_edge_index.numel() == 0:
            return 0, 0  # 或者其他合适的值

        roc_auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    return roc_auc, ap

# 训练主函数
def run_training(X_list, A_list, epochs=150, batch_size:int=25, lr=0.001):
    # 准备数据
    data_list = prepare_data(X=X_list, A=A_list)
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 动态创建模型（根据第一条数据的特征维度动态设置输入通道数）
    in_channels = 22
    hidden_channels = 32
    out_channels = 8

    # 实例化更新后的 DeepVGAE
    model = DeepVGAE(in_channels=in_channels, 
                     hidden_channels=hidden_channels, 
                     out_channels=out_channels).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    print('初始化完成 开始训练啦!')


    save_ruc =  []
    save_ap=    []
    save_loss = []
    # 训练和评估
    for epoch in range(epochs):
        total_loss = 0
        roc_auc_scores = []
        ap_scores = []

        for data  in loader:
            data = data.to(device)
            loss = train(model, data, optimizer)
            total_loss += loss

            # 测试模型
            roc_auc, ap = test(model, data)
            roc_auc_scores.append(roc_auc)
            ap_scores.append(ap)

        # 输出当前 epoch 的结果
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}')
        
        if (epoch + 1) % 10 == 0:
            # 保存模型和参数
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')
            print(f'Model saved at epoch {epoch+1}')

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}, Avg ROC AUC: {sum(roc_auc_scores)/len(roc_auc_scores):.4f}, Avg AP: {sum(ap_scores)/len(ap_scores):.4f}')
        
        save_ruc.append(sum(roc_auc_scores)/len(roc_auc_scores))
        save_ap.append(sum(ap_scores)/len(ap_scores))
        save_loss.append(total_loss / len(loader))
        outcome_dict={'AP':save_ap, 'ROC':save_ruc, 'Loss':save_loss} 
        with open('./result_1015.json', 'w') as f:
            json.dump(outcome_dict, f)


def save_model(model):
    torch.save(model.state_dict(), './Model_1015.pt')
    
    





# 加载数据（5000 对 (X, A)）
A_list = []
X_list = []
for i in range(5000):
    A = pd.read_csv(f'../VGAE_dataset/raw/A/A_{i}.csv', header=None).values
    A = A+A.T  # 确保对称性
    A_list.append(pd.read_csv(f'../VGAE_dataset/raw/A/A_{i}.csv', header=None).values)
    X_list.append(pd.read_csv(f'../VGAE_dataset/raw/X/X_{i}.csv', header=None).values)

# 运行训练
run_training(X_list, A_list)
