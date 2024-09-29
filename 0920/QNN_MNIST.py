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

'''
背景介绍 为了基于Mindspore 来复现GS-QAS,在训练pretext中的VGAE中 需要准备数据集: ANSATZ 与 其在MNIST数据集上的准确度
我们期待获得这样的数据: [Ansatz1 , 0.88],[Ansatz2, 0.89],[Ansatz3, 0.82]...
因此 本文件主要为了快速得到 输入Ansatz结构 对应的 判别准确度
但是老子用Mindspore 写一个随便的Ansatz例子 照理说起码训练后得到 75%以上准确度 但保持在60%左右 说明训练失败 
因此 你来帮我检查 为什么训练失败？

细节1: 数据准备阶段 我们只选取了3和6两个数字的数据集的20%数量 变成二分类问题 
所以标签处理: 我们将3替换为1 6替换为0 

细节2: MNIST 数字 是28*28 我们将其缩小为14*14 以减少量子比特数目 => 8量子比特

'''

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
        
        
#这是为了该死的数据加载 encoder部分
def amplitude_param(pixels,n_qbits:int=7):
    param_rd = []
    _, parameterResolver = amplitude_encoder(pixels, n_qbits)   
    for _, param in parameterResolver.items():
        param_rd.append(param)
    param_rd = np.array(param_rd)
    return param_rd

class MNIST_QNN():
    def __init__(self,zoom_factor:float = 0.4,hea_reps:int=1):
        
        self.zoom_factor = zoom_factor #(28,28)->(11,11) 降低比特数目
        self._dataset_init()
        self._n_qbits = 7
        self.simulator = Simulator('mqvector', self._n_qbits)
        self._circuit_init(hea_reps)
        self._train_init()
        
        

    def _dataset_init(self):
        
        '''
        此函数进行数据加载和处理 
        主要包括:1-挑选出3和6俩数字 2-进行数据降维 3-划分训练测试数据集 4-标签处理
        5-生成Mindspore的数据加载器
        
        '''
        print('正在进行数据预处理...稍等')
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
        X_data, y = filtered_data  # X 图像数据 y 标签
        
        X_data_3, y_data_3 = sample_data(X_data, y, label=3, sample_ratio=0.2)
        X_data_6, y_data_6 = sample_data(X_data, y, label=6, sample_ratio=0.2)
        
        X_sampled = torch.cat((X_data_3, X_data_6), dim=0)
        y_sampled = torch.cat((y_data_3, y_data_6), dim=0)
        
        Compressed_X = np.array([zoom(img,0.4) for img in X_sampled]) #数据降维 成（11*11）
        self.X = Compressed_X
        self.y = y_sampled
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(Compressed_X, y_sampled, test_size=0.2, random_state=0, shuffle=True)
        
        # 将值 3 替换为 1 ，将值 6 替换为 0 为了二分类
        self._y_train[self._y_train == 3] = 1
        self._y_train[self._y_train == 6] = 0
        
        self._y_test[self._y_test == 3] = 1
        self._y_test[self._y_test == 6] = 0
        
        
        self.train_params = np.array([amplitude_param(i.flatten()) for i in self._X_train])
        self.test_params  = np.array([amplitude_param(i.flatten()) for i in self._X_test])
        
        
        #数据集加载
        self.train_loader = NumpySlicesDataset({'features': self.train_params, 'labels': self._y_train}, shuffle=False).batch(10) 
        
        self.test_loader = NumpySlicesDataset({'features': self.test_params, 'labels': self._y_test}).batch(10)
        print(f'训练、测试数据集初始化完成!shape={self._X_train.shape}')
        

    def _circuit_init(self,reps:int):
        """
        设置Encoder Ansatz 量子神经网络
        以及输出用的测量算符 
         
        """
        
        encoder, parameterResolver  = amplitude_encoder(self._X_train[0].flatten(),n_qubits=self._n_qbits)
        encoder = encoder.no_grad() 
        encoder = encoder.as_encoder()
        
        ansatz = HardwareEfficientAnsatz(self._n_qbits, single_rot_gate_seq=[RY], entangle_gate=X, depth=reps).circuit
        #找两个比特来测量Z基底的期望值 
        hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [2, 3]]

        ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
        ms.set_seed(2)                                                     # 设置生成随机数的种子
        ansatz  = ansatz.as_ansatz()
        circuit = encoder + ansatz        
        grad_ops = self.simulator.get_expectation_with_grad(hams,
                                                circuit,
                                                parallel_worker=5)
        self.QuantumNet = MQLayer(grad_ops)          # 搭建量子神经网络
        
        
    def _train_init(self):
        Learning_rate = 0.05 #学习率 优化器参数 
        self.loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')          
        self.opti = Adam(self.QuantumNet.trainable_params(), learning_rate=Learning_rate)              
        self.model = Model(self.QuantumNet, self.loss, self.opti, metrics={'Acc': Accuracy()})  
    
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
    
        

