from mindquantum.algorithm.compiler import DAGCircuit
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.core.gates import RotPauliString
from mindquantum.core import gates
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.core.circuit import apply
from mindquantum.algorithm.compiler import  DAGCircuit,DAGQubitNode,DAGNode
import numpy as np
import networkx as nx
from copy import deepcopy
import torch
import pandas as pd
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix, train_test_split_edges

import scipy.sparse as sp


#将量子线路转化为两个矩阵 A｜X
def ConvertCircuit(circuit:Circuit,n_qbits:int=8):
    DAG_cir = DAGCircuit(circuit)
    All_node = DAG_cir.find_all_gate_node()
    Gate_Categary = []
    for each_node in All_node:
        if each_node.gate.name not in Gate_Categary:
            Gate_Categary.append(each_node.gate.name)
            
    # 按照首字母排序
    #Gate_Categary = sorted(Gate_Categary, key=lambda x: x.lower())
    Gate_Categary = ['X','Y','H','Z','RX','RY','RZ','Rxx','Ryy','Rzz','SWAP','CNOT']
    
    #X矩阵的列数 与 行数
    X_column_num = len(Gate_Categary) + 2 + n_qbits
    X_column_num = len(DAG_cir.layering()) + 2
    X_first_row = [1] + [0]*len(Gate_Categary) + [0] + [1]*n_qbits
    X_last_row =  [0] + [0]*len(Gate_Categary) + [1] + [1]*n_qbits
    X_middle_rows =[]
    
    for each_layer in DAG_cir.layering():
        each_layer_nodes = DAGCircuit(circuit=each_layer).find_all_gate_node()
        each_layer_nodes = sorted(each_layer_nodes, key=lambda x: x.gate.name.lower())
        for each_node in each_layer_nodes:
            index = Gate_Categary.index(each_node.gate.name)
            tmp_left = [0]*len(Gate_Categary) 
            tmp_left[index] = 1
            tmp_right = [0]*n_qbits
            tmp_right = [1 if i in each_node.gate.obj_qubits or i in each_node.gate.ctrl_qubits else 0 for i in range(n_qbits)]
            row = [0] + tmp_left + [0] + tmp_right
            X_middle_rows.append(row)        
    X_matrix = np.vstack([X_first_row] + X_middle_rows + [X_last_row])
    
    
    Gates = []
    Gates_name = []
    for each_layer in DAG_cir.layering():
        nodes = sorted(DAGCircuit(each_layer).find_all_gate_node(), key=lambda x: x.gate.name.lower())
        Gates.append(nodes)
        for node in nodes:
                Gates_name.append(node)
                
    G = nx.DiGraph()
    position = {}
    START_Node = "START"
    END_Node = "END"
    G.add_node(START_Node)
    G.add_node(END_Node)
    position[START_Node] = (-1, 0)
    position[END_Node] = (len(Gates)+1, 0)
    
    G.add_edges_from([[START_Node,i]for i in Gates[0]])
    G.add_edges_from([[i,END_Node]for i in Gates[-1]])
    
    
    for index,each_layer_node_a in enumerate(Gates):
        G.add_nodes_from(each_layer_node_a)
        for each_node in each_layer_node_a:
            position[each_node] = (index, each_node.gate.obj_qubits[0])
            position[each_node] = (index, each_node.gate.obj_qubits[0])
            obj_qubits = deepcopy(each_node.gate.obj_qubits)

            for layer in Gates[index+1:]:
                for each_layer_node_b in layer:
                    #判断是否有交集
                    if set(obj_qubits) & set(each_layer_node_b.gate.obj_qubits):
                        G.add_edge(each_node, each_layer_node_b)
                        intersection = list(set(obj_qubits) & set(each_layer_node_b.gate.obj_qubits))
                        [obj_qubits.remove(i) for i in intersection]
                        if len(obj_qubits) == 0:
                            break
                        else:
                            continue
        # 根据 position 中的顺序生成节点列表
    sorted_nodes = sorted(position.keys(), key=lambda x: position[x])

    # 生成邻接矩阵，确保顺序与 position 一致
    A_matrix = nx.adjacency_matrix(G, nodelist=sorted_nodes).todense()
    
    #A_matrix = nx.adjacency_matrix(G).todense()
    return X_matrix,A_matrix,G,position,Gates_name

    
def my_random_circuit(n_qubits, gate_num, sd_rate=0.5, ctrl_rate=0.2, seed=None):
    pg = PRGenerator(name='ansatz')
    if seed is None:
        seed = np.random.randint(1, 2**23)

    if n_qubits == 1:
        sd_rate = 1
        ctrl_rate = 0
    single = {
        'param': [gates.RX, gates.RY, gates.RZ],
        'non_param': [gates.X, gates.Y, gates.Z, gates.H],
    }
    double = {'param': [gates.Rxx, gates.Ryy, gates.Rzz], 'non_param': [gates.SWAP,gates.CNOT]}
    circuit = Circuit()
    np.random.seed(seed)
    qubits = range(n_qubits)
    for _ in range(gate_num):
        if n_qubits == 1:
            q1, q2 = int(qubits[0]), None
        else:
            q1, q2 = np.random.choice(qubits, 2, replace=False)
            q1, q2 = int(q1), int(q2)
        if np.random.random() < sd_rate:
            if np.random.random() > ctrl_rate:
                q2 = None
            if np.random.random() < 0.5:
                gate = np.random.choice(single['param'])
                #param = np.random.uniform(-np.pi * 2, np.pi * 2)
                param = pg.new()
                circuit += gate(param).on(q1)
            else:
                gate = np.random.choice(single['non_param'])
                circuit += gate.on(q1)
        else:
            if np.random.random() < 0.75:
                gate = np.random.choice(double['param'])
                #param = np.random.uniform(-np.pi * 2, np.pi * 2)
                param = pg.new()
                circuit += gate(param).on([q1, q2])
            else:
                gate = np.random.choice(double['non_param'])
                if gate is not gates.CNOT:
                    circuit += gate.on([q1, q2])
                else:
                    circuit += gate.on(q1, q2)
    return circuit
    
    


# 假设所有的邻接矩阵 A 和特征矩阵 X 都分别存储在不同的 CSV 文件中
# A 文件夹包含 5000 个 A 文件，例如 A_0.csv, A_1.csv, ... A_4999.csv
# X 文件夹包含 5000 个 X 文件，例如 X_0.csv, X_1.csv, ... X_4999.csv

class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        初始化 CustomGraphDataset 对象。

        参数:
        root (str): 数据集的根目录。
        transform (callable, optional): 用于数据转换的函数。
        pre_transform (callable, optional): 用于预处理数据的函数。
        """
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        返回原始数据文件的名称列表。
        因为我们直接处理 CSV 文件，所以这里留空。
        """
        return []

    @property
    def processed_file_names(self):
        """
        返回处理后的数据文件的名称列表。
        """
        return ['data.pt']

    def process(self):
        """
        处理原始数据并将其转换为 PyTorch Geometric 的 Data 对象。
        """
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

            # 分割边为训练和测试边
            data = train_test_split_edges(data)

            data_list.append(data)

        # 将所有图对象保存到单个文件中
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
def decoding_circuit(X:np.array):
    Gate_Categary = ['START','X','Y','H','Z','RX','RY','RZ','Rxx','Ryy','Rzz','SWAP','CNOT','END']
    cir = Circuit()
    row,col = X.shape
    pg = PRGenerator(name='ansatz')
    for i in range(row):
        row = X[i]
        node_index=row[:-8]
        qbits_index=row[-8:]
        if all(x == 0 for x in node_index):
            continue
        print(node_index)
        gate_str = Gate_Categary[np.where(node_index == 1)[0][0]]
        qbits = np.where(qbits_index == 1)[0]
        qbits = [int(qbit) for qbit in qbits]  # 将 numpy.int64 转换为 Python int
        print(gate_str,qbits)
        print(f'Gate ={gate_str}')

        if gate_str == 'START' or gate_str == 'END':
            continue
        if gate_str == 'X':
            if len(qbits)==1:
                cir +=gates.X.on(qbits[0])
            else:
                cir +=gates.X.on(qbits[0],qbits[1])
        if gate_str == 'Y':
            if len(qbits)==1:
                cir +=gates.Y.on(qbits[0])
            else:
                cir +=gates.Y.on(qbits[0],qbits[1])
                
        if gate_str == 'Z':
            if len(qbits)==1:
                cir +=gates.Z.on(qbits[0])
            else:
                cir +=gates.Z.on(qbits[0],qbits[1])
        
        if gate_str == 'H':
            if len(qbits)==1:
                cir +=gates.H.on(qbits[0])
            else:
                cir +=gates.H.on(qbits[0],qbits[1])
                
        if gate_str == 'RX':
            print('检测到 RX!')
            cir +=RX(pg.new()).on(qbits[0])
        if gate_str == 'RY':
            cir +=RY(pg.new()).on(qbits[0])
        if gate_str == 'RZ':
            cir +=RZ(pg.new()).on(qbits[0])
        if gate_str == 'CNOT':
            cir +=CNOT.on(qbits[0],qbits[1])
        if gate_str == 'SWAP':
            print(qbits[0],qbits[1])
            cir +=gates.SWAP.on(qbits)
        if gate_str == 'Rxx':
            cir +=gates.Rxx(pg.new()).on(qbits)
        if gate_str == 'Ryy':
            cir +=gates.Ryy(pg.new()).on(qbits)
        if gate_str == 'Rzz':
            cir +=gates.Rzz(pg.new()).on(qbits)
            
    return cir