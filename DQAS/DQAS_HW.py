import numpy as np
from mindquantum.core.gates import RX, RY, RZ, H, X, Y, Z, CNOT
from mindquantum.core.circuit import Circuit
import mindspore as ms
from mindquantum.simulator import  Simulator
from mindquantum.core.gates import GroupedPauli
from mindquantum.core.operators import TimeEvolution,QubitOperator
from mindquantum.core.parameterresolver import PRConvertible,PRGenerator,ParameterResolver
from DQAS_tool import generate_pauli_string,one_hot,unbound_opeartor_pool
from mindquantum.core.gates import RotPauliString
from mindquantum.core.gates import UnivMathGate
from mindspore import Tensor, ops
from mindquantum.core.circuit import UN
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
from mindquantum.framework import MQLayer
from mindspore.nn import  TrainOneStepCell
from mindspore.nn import SoftmaxCrossEntropyWithLogits                         # 导入SoftmaxCrossEntropyWithLogits模块，用于定义损失函数
from mindspore.nn import Adam                                                  # 导入Adam模块用于定义优化参数
from mindspore.train import Accuracy, Model, LossMonitor                       # 导入Accuracy模块，用于评估预测准确率
import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore.dataset import NumpySlicesDataset
from torch.utils.data import DataLoader# 导入NumpySlicesDataset模块，用于创建模型可以识别的数据集
import sys
sys.path.append('..')
from data_processing import X_train,X_test,y_train,y_test
from mindquantum.algorithm.nisq import HardwareEfficientAnsatz   

from DQAS_tool import Mindspore_ansatz,loss_fn,vag_nnp,sampling_from_structure,vag_nnp,vag_nnp_function,sampling_from_structure
from mindquantum.framework import MQOps
import mindspore.nn as nn
import numpy as np
import tensorcircuit as tc
import tensorflow as tf
  
num_layer = 6
# 定义标准差和形状
stddev = 0.02
shape_parametized = 12
shape_unparametized = 4
shape_nnp = (num_layer, shape_parametized)
shape_stp = (num_layer, shape_parametized+shape_unparametized)
shape_stp = (num_layer, shape_parametized)
# 使用 numpy 生成随机数矩阵
np.random.seed(10)
nnp = np.random.normal(loc=0.0, scale=stddev, size=shape_nnp).astype(rtype)
stp = np.random.normal(loc=0.0, scale=stddev, size=shape_stp).astype(rtype)
# #Operator Pool
unbound_opeartor_pool = [generate_pauli_string(n=8,seed=i)[0] for i in range(shape_parametized)]
bound_opeartor_pool = [generate_pauli_string(n=8,seed=i)[1] for i in range(shape_parametized,shape_parametized+shape_unparametized)]


class Mindspore_DQAS():
    def __init__(self,Structure_params:np.array,Ansatz_params:np.array,num_layer:int,nqbits:int=8):
        
        self.num_layer = Structure_params.shape[0]
        
        pass