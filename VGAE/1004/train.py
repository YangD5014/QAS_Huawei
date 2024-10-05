# from Pytorch_VGAE import CustomGraphDataset,VGAEModel
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
from torch_geometric.loader import DataLoader
import numpy as np
from VGAE import VGAEEncoder,prepare_data
from torch_geometric.nn import VGAE
import torch


A_list = []
X_list = []
for i in range(5000):
    A = pd.read_csv(f'../../VGAE_dataset/raw/A/A_{i}.csv', header=None).values
    A = A+A.T
    A_list.append(A)
    X_list.append(pd.read_csv(f'../../VGAE_dataset/raw/X/X_{i}.csv', header=None).values)
    
data_list = prepare_data(X_list, A_list)
loader = DataLoader(data_list, batch_size=1, shuffle=True)

in_channels = 22
hidden_channels = 64
out_channels = 16
num_layers = 3
epochs = 200
batch_size = 1
lr = 0.001


def train():
    model.train()  # 设置模型为训练模式
    total_loss = 0  # 新增：记录总损失
    for batch in loader:  # 批量处理
        optimizer.zero_grad()
        batch = batch.to(device)
        z = model.encode(batch.x, batch.train_pos_edge_index)  # 对 batch 进行编码
        # 计算重构损失和 KL 损失
        loss = model.recon_loss(z, batch.train_pos_edge_index)
        loss += (1 / batch.num_nodes) * model.kl_loss()  # 使用 batch.num_nodes 归一化 KL 损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()  # 累加每个批次的损失
    return total_loss / len(loader)  # 返回平均损失  


def test():
    #print('正在评估中...🀄️')
    model.eval()  # 设置模型为评估模式
    total_auc = 0  # 记录总 AUC
    total_ap = 0   # 记录总 AP
    valid_batches = 0  # 记录有效批次数量

    with torch.no_grad():  # 禁用梯度计算
        for batch in loader:  # 对每个 batch 进行测试
            batch = batch.to(device)

            # 检查 test_pos_edge_index 和 test_neg_edge_index 是否为空
            if batch.test_pos_edge_index.size(1) == 0 or batch.test_neg_edge_index.size(1) == 0:
                # print("警告: test_pos_edge_index 或 test_neg_edge_index 为空，跳过该批次。")
                continue  # 跳过当前批次

            z = model.encode(batch.x, batch.train_pos_edge_index)  # 对 batch 进行编码
            
            # 使用测试集的正边和负边进行测试
            auc, ap = model.test(z, batch.test_pos_edge_index, batch.test_neg_edge_index)
            total_auc += auc
            total_ap += ap
            valid_batches += 1  # 增加有效批次计数

    # 返回平均 AUC 和 AP，如果没有有效批次，则返回 0
    return (total_auc / valid_batches) if valid_batches > 0 else 0.0, (total_ap / valid_batches) if valid_batches > 0 else 0.0





if __name__ == '__main__':
    
    model = VGAE(encoder=VGAEEncoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
        
    # 主训练循环
    for epoch in range(1, epochs + 1):
        loss = train()  # 训练
        auc, ap = test()  # 测试
        print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, loss, auc, ap))
        
        # 每 10 轮保存一次模型权重
        if epoch % 10 == 0:
            model_filename = 'model_epoch_{:03d}_AUC_{:.4f}_AP_{:.4f}.pt'.format(epoch, auc, ap)
            torch.save(model.state_dict(), model_filename)
            print(f'Model saved at epoch {epoch} with AUC: {auc:.4f} and AP: {ap:.4f}')




