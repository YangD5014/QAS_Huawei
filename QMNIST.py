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
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz     
from mindquantum.core.operators import QubitOperator           # 导入QubitOperator模块，用于构造泡利算符
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
import mindspore as ms                                                                         # 导入mindspore库并简写为ms
from mindquantum.framework import MQLayer                                                      # 导入MQLayer
# 导入HardwareEfficientAnsatz
from mindquantum.core.gates import RY           
import torch
from torchvision import datasets, transforms# 导入量子门RY
from scipy.ndimage import zoom
import random

from mindspore.nn import SoftmaxCrossEntropyWithLogits                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam                                                  # 导入Adam模块用于定义优化参数
from mindspore.train import Accuracy, Model, LossMonitor                       # 导入Accuracy模块，用于评估预测准确率
import mindspore as ms
from mindspore.dataset import NumpySlicesDataset
from torch.utils.data import DataLoader# 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集


np.random.seed(10)

def sample_data(X, y, label, sample_ratio=0.2):
    label_mask = (y == label)
    X_label = X[label_mask]
    y_label = y[label_mask]
    
    sample_size = int(len(y_label) * sample_ratio)
    sample_indices = np.random.choice(len(y_label), sample_size, replace=False)
    
    return X_label[sample_indices], y_label[sample_indices]

def filter_3_and_6(data):
    images, labels = data
    mask = (labels == 3) | (labels == 6)
    return images[mask], labels[mask]
        

class MNIST_QNN():
    
    def __init__(self,zoom_factor:float = 0.4,hea_reps:int=1):
        
        self.zoom_factor = zoom_factor #(28,28)->(11,11) 降低比特数目
        self._dataset_init()
        self._n_qbits = 6
        self.simulator = Simulator('mqvector', self._n_qbits)
        self._circuit_init(hea_reps)
        self._train_init()
        
        
    def _dataset_init(self):
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
        X_data, y = filtered_data  # X 图像数据 y 标签
        
        X_data_3, y_data_3 = sample_data(X_data, y, label=3, sample_ratio=0.2)
        X_data_6, y_data_6 = sample_data(X_data, y, label=6, sample_ratio=0.2)
        
        X_sampled = torch.cat((X_data_3, X_data_6), dim=0)
        y_sampled = torch.cat((y_data_3, y_data_6), dim=0)
        
        Compressed_X = np.array([zoom(img,0.4) for img in X_sampled])
        self.X = Compressed_X
        self.y = y_sampled
        
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(Compressed_X, y_sampled, test_size=0.2, random_state=0, shuffle=True)
        
        
        # 将值 3 替换为 1 ，将值 6 替换为 0
        self._y_train[self._y_train == 3] = 1
        self._y_train[self._y_train == 6] = 0
        
        self._y_test[self._y_test == 3] = 1
        self._y_train[self._y_train == 6] = 0
        print(f'训练、测试数据集初始化完成!shape={self._X_train.shape}')
        

    def _circuit_init(self,reps:int):
        # prg = PRGenerator('alpha')
        # # encoder = Circuit()
        encoder, parameterResolver  = amplitude_encoder(self._X_train[0].flatten(),n_qubits=self._n_qbits)
        encoder = encoder.no_grad() 
        ansatz = HardwareEfficientAnsatz(self._n_qbits, single_rot_gate_seq=[RY], entangle_gate=X, depth=reps).circuit     # 通过
        hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]

        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        ms.set_seed(1)                                                     # 设置生成随机数的种子
        encoder = encoder.as_encoder()
        ansatz  = ansatz.as_ansatz()
        circuit = encoder.as_encoder() + ansatz.as_ansatz()         
        grad_ops = self.simulator.get_expectation_with_grad(hams,
                                                circuit,
                                                parallel_worker=5)
        self.QuantumNet = MQLayer(grad_ops)          # 搭建量子神经网络
        
        
    def _train_init(self):
        self.loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')          
        self.opti = Adam(self.QuantumNet.trainable_params(), learning_rate=0.1)              
        self.model = Model(self.QuantumNet, self.loss, self.opti, metrics={'Acc': Accuracy()})  
        
        X_train_for_loader = self._X_train.reshape(-1,11*11)
        padding_size = 126 - X_train_for_loader.shape[1]
        X_train_for_loader = np.pad(X_train_for_loader, ((0, 0), (0, padding_size)), mode='constant')
        
        X_test_for_loader = self._X_test.reshape(-1,11*11)
        padding_size = 126 - X_train_for_loader.shape[1]
        X_train_for_loader = np.pad(X_train_for_loader, ((0, 0), (0, padding_size)), mode='constant')
        
        self.train_loader = NumpySlicesDataset({'features': X_train_for_loader, 'labels': self._y_train}, shuffle=False).batch(10) # 通过NumpySlicesDataset创建训练样本的数据集，shuffle=False表示不打乱数据，batch(5)表示训练集每批次样本点有5个
        self.test_loader = NumpySlicesDataset({'features': X_test_for_loader, 'labels': self._y_test}).batch(10)
        
    
class StepAcc(ms.Callback):                                                      # 定义一个关于每一步准确率的回调函数
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def on_train_step_end(self, run_context):
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])    
                

if __name__ == '__main__':
    mnist_qnn = MNIST_QNN(hea_reps=1)
    monitor = LossMonitor(20)                                                      
    acc = StepAcc(mnist_qnn.model, mnist_qnn.test_loader)  
    mnist_qnn.model.train(10, mnist_qnn.train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)
    
        

