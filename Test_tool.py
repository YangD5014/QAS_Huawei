import numpy as np                                          
from mindquantum.core.circuit import Circuit                
from mindquantum.core.gates import H, RX, RY, RZ,X    
from mindquantum.core.parameterresolver import PRGenerator  
from mindquantum.simulator import Simulator
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split   
from mindquantum.algorithm.library import amplitude_encoder
from mindquantum.algorithm.nisq import IQPEncoding
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz     
from mindquantum.core.operators import QubitOperator           # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
import mindspore as ms                                                                         # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer,MQN2Layer                                              # 导入MQLayer
# 导入HardwareEfficientAnsatz
from mindquantum.core.gates import RY           
from mindquantum.core.circuit import UN
import torch
from torchvision import datasets, transforms# 导入量子门RY
from scipy.ndimage import zoom
import random
from data_processing import PCA_data_preprocessing
from mindspore.nn import SoftmaxCrossEntropyWithLogits                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam                                                  # 导入Adam模块用于定义优化参数
from mindspore.train import Accuracy, Model, LossMonitor                       # 导入Accuracy模块，用于评估预测准确率
import mindspore as ms
from mindspore.dataset import NumpySlicesDataset
from torch.utils.data import DataLoader# 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集



class StepAcc(ms.Callback):                                                      # 定义一个关于每一步准确率的回调函数
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def on_train_step_end(self, run_context):
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])
        # print(f'ACC = {self.acc[-1]}')


def Test_ansatz(ansatz:Circuit=None,learning_rate=0.01,epochs:int=15):
    # 下载和加载 MNIST 数据集
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    X_train, X_test, y_train, y_test = PCA_data_preprocessing(mnist_dataset,8)
    
    prg = PRGenerator('alpha')
    nqbits = 8
    encoder = Circuit()
    encoder += UN(H, nqbits)                                  # H门作用在每1位量子比特
    for i in range(nqbits):                                   # i = 0, 1, 2, 3
        encoder += RY(prg.new()).on(i)                 # RZ(alpha_i)门作用在第i位量子比特
    encoder = encoder.no_grad()
    encoder = encoder.as_encoder()
    
    if ansatz is None:
        ansatz = HardwareEfficientAnsatz(nqbits, single_rot_gate_seq=[RY], entangle_gate=X, depth=2).circuit     # 通过
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0,1]]
    ansatz = ansatz.as_ansatz()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_seed(2)                                                     # 设置生成随机数的种子
    circuit = encoder+ ansatz.as_ansatz()         
    sim = Simulator('mqvector', n_qubits=nqbits)
    grad_ops = sim.get_expectation_with_grad(hams,
                                            circuit,
                                            parallel_worker=5)
    QuantumNet = MQLayer(grad_ops)          # 搭建量子神经网络
    
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')            
    opti = Adam(QuantumNet.trainable_params(), learning_rate=learning_rate)                  
    model = Model(QuantumNet, loss, opti, metrics={'Acc': Accuracy()})             
    train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(20) 
    test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test},shuffle=False).batch(20)                   
    
    monitor = LossMonitor(20)                                                     
    acc = StepAcc(model, test_loader)                                 
    model.train(epochs, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)
    return acc.acc
    