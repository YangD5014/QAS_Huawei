import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.sparse as sp
import numpy as np
import os
import pandas as pd
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data, DataLoader
from GVAE_GIN import DeepVGAE

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

# 验证模型函数
def validate(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)

        # 添加检查
        if data.val_pos_edge_index.numel() == 0 or data.val_neg_edge_index.numel() == 0:
            return 0, 0  # 或者其他合适的值

        roc_auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
    return roc_auc, ap

# 训练主函数
def run_training(X_list, A_list, epochs=200, batch_size:int=1, lr=0.001):
    # 准备数据
    data_list = prepare_data(X=X_list, A=A_list)
    
    # 使用 RandomLinkSplit 进行训练/验证/测试分割
    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True)
    
    # 分割数据
    split_data_list = []
    val_data_list = []
    test_data_list = []
    for data in data_list:
        train_data, val_data, test_data = transform(data)
        #train_data = 
        val_data_list.append(val_data)
        test_data_list.append(test_data)
        
        split_data_list.append(train_data)  # 只使用训练数据进行训练
    
    train_loader = DataLoader(split_data_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)
    
    
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 动态创建模型（根据第一条数据的特征维度动态设置输入通道数）
    in_channels = 22
    hidden_channels = 32
    out_channels = 12

    # 实例化更新后的 DeepVGAE
    model = DeepVGAE(in_channels=in_channels, 
                     hidden_channels=hidden_channels, 
                     out_channels=out_channels).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    print('初始化完成 开始训练啦!')

    # 训练和评估
    for epoch in range(epochs):
        total_loss = 0
        roc_auc_scores = []
        ap_scores = []
        
        for data in train_loader:
            data = data.to(device)
            loss = train(model, data, optimizer)
            total_loss += loss

        # 在每个 epoch 后进行验证
        val_roc_auc_scores = []
        val_ap_scores = []
        for data in val_loader:
            data = data.to(device)
            roc_auc, ap = validate(model, data)
            val_roc_auc_scores.append(roc_auc)
            val_ap_scores.append(ap)

        # 测试模型
        test_roc_auc_scores = []
        test_ap_scores = []
        for data in test_loader:
            data = data.to(device)
            roc_auc, ap = test(model, data)
            test_roc_auc_scores.append(roc_auc)
            test_ap_scores.append(ap)

        # 输出当前 epoch 的结果
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}, '
              f'Val Avg ROC AUC: {sum(val_roc_auc_scores) / len(val_roc_auc_scores):.4f}, '
              f'Val Avg AP: {sum(val_ap_scores) / len(val_ap_scores):.4f}, '
              f'Test Avg ROC AUC: {sum(test_roc_auc_scores) / len(test_roc_auc_scores):.4f}, '
              f'Test Avg AP: {sum(test_ap_scores) / len(test_ap_scores):.4f}')

        if (epoch + 1) % 10 == 0:
            # 保存模型和参数
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')
            print(f'Model saved at epoch {epoch+1}')


# 加载数据（5000 对 (X, A)）
A_list = []
X_list = []
for i in range(5000):
    A = pd.read_csv(f'../VGAE_dataset/raw/A/A_{i}.csv', header=None).values
    A = A + A.T  # 确保对称性
    A_list.append(A)
    X_list.append(pd.read_csv(f'../VGAE_dataset/raw/X/X_{i}.csv', header=None).values)

# 运行训练
run_training(X_list, A_list)
