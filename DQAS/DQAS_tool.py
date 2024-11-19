import numpy as np
from mindquantum.core.gates import RX, RY, RZ, H, X, Y, Z, CNOT
from mindquantum.core.circuit import Circuit,UN
import mindspore as ms
from mindquantum.simulator import  Simulator
from mindquantum.core.gates import GroupedPauli
from mindquantum.core.operators import TimeEvolution,QubitOperator
from mindquantum.core.parameterresolver import PRGenerator
from mindspore.nn import Adam  
import random
from mindspore import Tensor,ops,Parameter
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
from mindquantum.framework import MQLayer,MQOps
from torchvision import datasets
import mindspore.numpy as mnp
import sys
from typing import Union
sys.path.append('..')
from Test_tool import Test_ansatz
from data_processing import X_train,X_test,y_train,y_test
from mindspore.train import Accuracy, Model, LossMonitor  
from mindspore.dataset import NumpySlicesDataset
from mindquantum.core.circuit import change_param_name,apply



def generate_pauli_string(n, seed=None):
    pauli_operators = ['X', 'Y', 'Z', 'I']
    pauli_string = []
    
    pauli_string_without_num = []
    if seed is not None:
        random.seed(seed)
    for i in range(n):
        operator = random.choice(pauli_operators)
        pauli_string.append(f"{operator}{i}")
        pauli_string_without_num.append(f"{operator}")
    
    return " ".join(pauli_string),"".join(pauli_string_without_num)

loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean') 

def one_hot(labels, num_classes):
    one_hot_op = ops.OneHot()
    on_value = Tensor(1.0, ms.int32)
    off_value = Tensor(0.0, ms.int32)
    return one_hot_op(labels, num_classes, on_value, off_value)




def Mindspore_ansatz(Structure_p:np.array,parameterized_pool:list,unparameterized_pool:list,num_layer:int=6,n_qbits:int=8):
    """
    和 DQAS 文章描述的一致，生成权重线路
    Structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    
    """
    if Structure_p.shape[0] != num_layer:
        raise ValueError('Structure_p shape must be equal to num_layer')
    softmax = ops.Softmax()
    my_stp = softmax(Tensor(Structure_p, ms.float32))
    
    prg = PRGenerator('encoder')
    nqbits = n_qbits
    encoder = Circuit()
    encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)                 
        
    ansatz = Circuit()
    pr_gen = PRGenerator('ansatz')
    #print(my_stp.shape)
    for i in range(num_layer):
        paramertized_part_count=0
        for index_op,each_op in enumerate(parameterized_pool):
            #print(my_stp[i,index_op])
            ansatz += TimeEvolution(QubitOperator(terms=each_op,coefficient=pr_gen.new()),time=float(my_stp[i,index_op])).circuit
            paramertized_part_count+=1
            
        for index_op,each_op in enumerate(unparameterized_pool):
            op = GroupedPauli(each_op)
            tmp_cir = Circuit([GroupedPauli(each_op).on(range(n_qbits))])
            matrix = tmp_cir.matrix()
            ansatz += UnivMathGate(matrix_value=matrix*float(my_stp[i,index_op+paramertized_part_count]),name=f'{my_stp[i,index_op+paramertized_part_count]}*{op.pauli_string}').on(range(n_qbits))  
    
    finnal_ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    return finnal_ansatz
    



def vag_nnp(Structure_params: np.array, Ansatz_params: np.array,paramerterized_pool:list,num_layer:int=6,n_qbits:int=8):
    """
    用于计算梯度 Ansatz_params关于 loss 的梯度
    value,grad_ansatz_params = vag(训练数据,标签数据)
    """
    ansatz = Mindspore_ansatz(Structure_p=Structure_params,parameterized_pool=paramerterized_pool,
                              num_layer=num_layer,n_qbits=n_qbits)
    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    #print(Ansatz_params.shape)
    Mylayer = MQLayer(grad_ops,ms.Tensor(Ansatz_params,ms.float64).reshape(-1))


    def forward_fn(encode_p,y_label):
        eval_obserables = Mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss
    # nnp = ms.Tensor(Ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn,grad_position=None,weights=Mylayer.trainable_params())
    
    return grad_fn


# def vag_nnp_function(Structure_params,Ansatz_params,X_train, y_train):
#     grad_fn = vag_nnp(Structure_params, Ansatz_params, n_layer=3, n_qbits=8)
#     loss, grads = grad_fn(ms.Tensor(X_train), ms.Tensor(y_train))
#     return loss, grads


def sampling_from_structure(structures: np.array, num_layer: int, shape_parametized: int) -> np.array:
    softmax = ops.Softmax()
    prob = softmax(ms.Tensor(structures, ms.float32))
    prob_np = prob.asnumpy()  # 将 MindSpore Tensor 转换为 NumPy 数组

    while True:
        samples = []
        for i in range(num_layer):
            sample = np.random.choice(prob_np[i].shape[0], p=prob_np[i])
            samples.append(sample)
        
        # 判断是否元素全都大于 shape_parametized
        if all(sample >= shape_parametized for sample in samples):
            continue  # 如果是，就重新采样
        else:
            break  # 如果不是，跳出循环

    return np.array(samples)


def DQASAnsatz_from_result(best_candidate:np.array,parameterized_pool:list,num_layer:int=6,n_qbits:int=8):
    prg = PRGenerator('encoder')
    if best_candidate.shape[0] != num_layer:
        raise ValueError('best_candidate shape must be equal to num_layer')
    
    nqbits = n_qbits
    encoder = Circuit()
    encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)     
    
    ansatz = Circuit()
    pr_gen = PRGenerator('ansatz')            
    for index_op,each_op in enumerate(best_candidate):
        ansatz += TimeEvolution(QubitOperator(terms=parameterized_pool[each_op],coefficient=pr_gen.new()),time=1).circuit
    
    ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    acc = Test_ansatz(ansatz)
    return ansatz,acc

def DQAS_accuracy(ansatz: Circuit,Network_params:np.array,n_qbits:int=8):
    
    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    op = MQOps(grad_ops)
    raw_result = op(ms.Tensor(X_train),ms.Tensor(Network_params).reshape(-1))
    softmax_pro = ops.Softmax()(raw_result)
    predicted_result= ops.Argmax()(softmax_pro)
    equal_elements = ops.equal(ms.Tensor(y_train),predicted_result)
    num_equal_elements = ops.reduce_sum(equal_elements.astype(ms.int32))
    acc = num_equal_elements.asnumpy()/X_train.shape[0]
    return acc

    
    
    
    


def nmf_gradient(structures:np.array, oh:ms.Tensor,num_layer: int,size_pool:int):
    """
    使用 MindSpore 实现蒙特卡洛梯度计算。
    """
      # Step 1: 获取选择的索引
    choice = ops.Argmax(axis=-1)(oh)
    # Step 2: 计算概率
    softmax = ops.Softmax(axis=-1)
    prob = softmax(ms.Tensor(structures))
    # Step 3: 获取概率矩阵中的值
    indices = mnp.stack((mnp.arange(num_layer, dtype=ms.int64), choice), axis=1)
    prob = ops.GatherNd()(prob, indices)
    # Step 4: 变换概率矩阵
    prob = prob.reshape(-1, 1)
    prob = ops.Tile()(prob, (1, size_pool))
    
    # Step 5: 生成蒙特卡洛梯度
    gradient = ops.TensorScatterAdd()(Tensor(-prob, ms.float64), indices, mnp.ones((num_layer,), dtype=ms.float64))
    return gradient
    
    
# 对向量化版本的封装
# nmf_gradient_vmap = ops.vmap(nmf_gradient, in_axes=(None, 0, None, None))

def best_from_structure(structures: np.array)->Tensor:
    return ops.Argmax(axis=-1)(ms.Tensor(structures))


def Mindspore_ansatz2(Structure_p:np.array,
                     parameterized_pool:list,
                     unparameterized_pool:list,
                     num_layer:int=6,
                     n_qbits:int=8):
    """
    和 DQAS 文章描述的一致，生成权重线路
    更新了非参数化门的算符池引入
    Structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    
    """
    if Structure_p.shape[0] != num_layer:
        raise ValueError('Structure_p shape must be equal to num_layer')
    
    if Structure_p.shape[1] != len(parameterized_pool)+len(unparameterized_pool):
        raise ValueError('Structure_p shape must be equal to size of pool')
    # softmax = ops.Softmax()
    # my_stp = softmax(Tensor(Structure_p, ms.float32))
    if isinstance(Structure_p, np.ndarray):
        my_stp = ms.Tensor(Structure_p, ms.float32)
    else:
        my_stp = Structure_p
        
    prg = PRGenerator('encoder')
    nqbits = n_qbits
    encoder = Circuit()
    # encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)                 
        
    ansatz = Circuit()
    pr_gen = PRGenerator('ansatz')
    #print(my_stp.shape)
    for layer_index in range(my_stp.shape[0]):
        for op_index in range(my_stp.shape[1]):
            if my_stp[layer_index,op_index] == 0:
                continue
            if op_index < len(parameterized_pool):
                ansatz += TimeEvolution(QubitOperator(terms=parameterized_pool[op_index],coefficient=pr_gen.new()),time=float(my_stp[layer_index,op_index])).circuit
            else:
                op = GroupedPauli(unparameterized_pool[op_index-len(parameterized_pool)])
                tmp_cir = Circuit([GroupedPauli(unparameterized_pool[op_index-len(parameterized_pool)]).on(range(n_qbits))])
                matrix = tmp_cir.matrix()
                ansatz += UnivMathGate(matrix_value=matrix*float(my_stp[layer_index,op_index]),name=f'{my_stp[layer_index,op_index]}*{op.pauli_string}').on(range(n_qbits))
    
    finnal_ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    return finnal_ansatz



def vag_nnp2(Structure_params: np.array, Ansatz_params: np.array,paramerterized_pool:list,unparamerterized_pool:list,num_layer:int=6,n_qbits:int=8):
    """
    更新: 只在对应位置上更新nnp梯度
    用于计算梯度 Ansatz_params关于 loss 的梯度
    value,grad_ansatz_params = vag(训练数据,标签数据)
    """
    if isinstance(Structure_params, np.ndarray):
        mystp= ms.Tensor(Structure_params,ms.float32)
    else:
        mystp = Structure_params
    ansatz = Mindspore_ansatz2(Structure_p=mystp,parameterized_pool=paramerterized_pool,
                               unparameterized_pool=unparamerterized_pool,
                              num_layer=num_layer,n_qbits=n_qbits)
    
    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    op_index = [ops.Argmax()(i) for i in mystp]
    #print(f'op_index={op_index}')
    ansatz_parameters=[]
    for layerIndex,i in enumerate(op_index):
        if i >=len(paramerterized_pool):
            continue
        else:
            ansatz_parameters.append(Ansatz_params[layerIndex,i])
    #print(f'ansatz_parameters={ansatz_parameters}')
    Mylayer = MQLayer(grad_ops,ms.Tensor(ansatz_parameters,ms.float64).reshape(-1))


    def forward_fn(encode_p,y_label):
        eval_obserables = Mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss
    # nnp = ms.Tensor(Ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn,grad_position=None,weights=Mylayer.trainable_params())
    
    return grad_fn


def vag_nnp3(Structure_params: np.array, Ansatz_params: np.array,paramerterized_pool:list[Circuit],unparamerterized_pool:list[Circuit],num_layer:int=6,n_qbits:int=8):
    """
    更新: 只在对应位置上更新nnp梯度
    用于计算梯度 Ansatz_params关于 loss 的梯度
    value,grad_ansatz_params = vag(训练数据,标签数据)
    """
    if isinstance(Structure_params, np.ndarray):
        mystp= ms.Tensor(Structure_params,ms.float32)
    else:
        mystp = Structure_params
    ansatz = Mindspore_ansatz3(Structure_p=mystp,parameterized_pool=paramerterized_pool,
                               unparameterized_pool=unparamerterized_pool,
                              num_layer=num_layer,n_qbits=n_qbits)
    
    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    op_index = [ops.Argmax()(i) for i in mystp]
    #print(f'op_index={op_index}')
    ansatz_parameters=[]
    for layerIndex,i in enumerate(op_index):
        if i >=len(paramerterized_pool):
            continue
        else:
            ansatz_parameters.append(Ansatz_params[layerIndex,i])
    Mylayer = MQLayer(grad_ops,ms.Tensor(ansatz_parameters,ms.float64).reshape(-1))


    def forward_fn(encode_p,y_label):
        eval_obserables = Mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss
    # nnp = ms.Tensor(Ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn,grad_position=None,weights=Mylayer.trainable_params())
    
    return grad_fn




def zeroslike_grad_nnp(batch_sturcture: Union[np.ndarray, ms.Tensor], grad_nnp: Union[np.ndarray, ms.Tensor], shape_parametized: int, ansatz_parameters: np.ndarray) -> np.ndarray:
    '''
    用于根据算出的梯度更新ansatz参数    
    '''
    if isinstance(batch_sturcture, np.ndarray):
        mystp = ms.Tensor(batch_sturcture, ms.float32)
    else:
        mystp = batch_sturcture  # 如果 batch_sturcture 已经是 ms.Tensor 类型

    op_index = [ops.Argmax()(i) for i in mystp]
    zeros_grad_nnp = np.zeros_like(ansatz_parameters)
    print(zeros_grad_nnp.shape)
    count = 0
    for each_sub in range(7):
        for layer, i in enumerate(op_index):
            if i >= shape_parametized:
                continue
            print(each_sub,layer,i)
            zeros_grad_nnp[each_sub,layer,i] = grad_nnp[count]
            count += 1
        
    return zeros_grad_nnp


        
def extract_parameterss(Structure_parameters:np.array,Candidate_index:np.array,shape_parametized:int):
    
    '''
    根据 候选index 从共享参数池中获取ansatz参数
    '''
    ansatz_parameters=[]
    for layerindex,i in enumerate(Candidate_index):
        if i >= shape_parametized:
            continue
        ansatz_parameters.append(Structure_parameters[layerindex,i])
        
    return ansatz_parameters




def Mindspore_ansatz3(Structure_p:np.array,
                     parameterized_pool:list[Circuit],
                     unparameterized_pool:list[Circuit],
                     num_layer:int=6,
                     n_qbits:int=8):
    """
    和 DQAS 文章描述的一致，生成权重线路
    更新了非参数化门的算符池引入
    Structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    
    """
    if Structure_p.shape[0] != num_layer:
        raise ValueError('Structure_p shape must be equal to num_layer')
    
    if Structure_p.shape[1] != len(parameterized_pool)+len(unparameterized_pool):
        raise ValueError('Structure_p shape must be equal to size of pool')
    if isinstance(Structure_p, np.ndarray):
        my_stp = ms.Tensor(Structure_p, ms.float32)
    else:
        my_stp = Structure_p
        
    prg = PRGenerator('encoder')
    nqbits = n_qbits
    encoder = Circuit()
    encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)                 
        
    ansatz = Circuit()
    pr_gen = PRGenerator('ansatz')
    #print(my_stp.shape)
    for layer_index in range(my_stp.shape[0]):
        for op_index in range(my_stp.shape[1]):
            if my_stp[layer_index,op_index] == 0:
                continue
            if op_index < len(parameterized_pool):
                ansatz += TimeEvolution(QubitOperator(terms=parameterized_pool[op_index],coefficient=pr_gen.new()),time=float(my_stp[layer_index,op_index])).circuit
            else:
                op = GroupedPauli(unparameterized_pool[op_index-len(parameterized_pool)])
                tmp_cir = Circuit([GroupedPauli(unparameterized_pool[op_index-len(parameterized_pool)]).on(range(n_qbits))])
                matrix = tmp_cir.matrix()
                ansatz += UnivMathGate(matrix_value=matrix*float(my_stp[layer_index,op_index]),name=f'{my_stp[layer_index,op_index]}*{op.pauli_string}').on(range(n_qbits))
    
    finnal_ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    return finnal_ansatz

class StepAcc(ms.Callback):                                                      # 定义一个关于每一步准确率的回调函数
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.acc = []

    def on_train_step_end(self, run_context):
        self.acc.append(self.model.eval(self.test_loader, dataset_sink_mode=False)['Acc'])
        # print(f'ACC = {self.acc[-1]}')

def Test_ansatz(ansatz:Circuit,learning_rate=0.01,epochs:int=15):
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0,1]]
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    ms.set_seed(2)                                                     # 设置生成随机数的种子  
    sim = Simulator('mqvector', n_qubits=8)
    grad_ops = sim.get_expectation_with_grad(hams,
                                            ansatz,
                                            parallel_worker=5)
    QuantumNet = MQLayer(grad_ops)          # 搭建量子神经网络     
    opti = Adam(QuantumNet.trainable_params(), learning_rate=learning_rate)                  
    model = Model(QuantumNet, loss_fn, opti, metrics={'Acc': Accuracy()})        
    batch_size =100     
    train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(batch_size) 
    test_loader = NumpySlicesDataset({'features': X_test, 'labels': y_test},shuffle=False).batch(batch_size)                   
    
    monitor = LossMonitor(20)                                                     
    acc = StepAcc(model, test_loader)                                 
    model.train(epochs, train_loader, callbacks=[monitor, acc], dataset_sink_mode=False)
    return acc.acc
    
    
def Mindspore_ansatz3(Structure_p:np.array,
                     parameterized_pool:list[Circuit],
                     unparameterized_pool:list[Circuit],
                     num_layer:int=6,
                     n_qbits:int=8):
    """
    和 DQAS 文章描述的一致，生成权重线路
    更新了非参数化门的算符池引入
    Structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    
    """
    if Structure_p.shape[0] != num_layer:
        raise ValueError('Structure_p shape must be equal to num_layer')
    
    if Structure_p.shape[1] != len(parameterized_pool)+len(unparameterized_pool):
        raise ValueError('Structure_p shape must be equal to size of pool')
    # softmax = ops.Softmax()
    # my_stp = softmax(Tensor(Structure_p, ms.float32))
    if isinstance(Structure_p, np.ndarray):
        my_stp = ms.Tensor(Structure_p, ms.float32)
    else:
        my_stp = Structure_p
        
    prg = PRGenerator('encoder')
    nqbits = n_qbits
    encoder = Circuit()
    # encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)    
    encoder = encoder.as_encoder()             
        
    ansatz = Circuit()
    pr_gen = PRGenerator('ansatz')
    #print(my_stp.shape)
    for layer_index in range(my_stp.shape[0]):
        for op_index in range(my_stp.shape[1]):
            if my_stp[layer_index,op_index] == 0:
                continue
            if op_index < len(parameterized_pool):
                before_ansatz = parameterized_pool[op_index]
                before_ansatz = change_param_name(circuit_fn=before_ansatz,name_map={before_ansatz.ansatz_params_name[0]:f'ansatz{layer_index}'})
                ansatz += before_ansatz
            else:
                ansatz += unparameterized_pool[op_index-len(parameterized_pool)]
    
    
    
    finnal_ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    # print(finnal_ansatz)
    # name_map = dict(zip(finnal_ansatz.ansatz_params_name,[f'ansatz{i}'for i in range(len(finnal_ansatz.ansatz_params_name))]))
    # finnal_ansatz = change_param_name(circuit_fn=finnal_ansatz,name_map=name_map)
    return finnal_ansatz

def wash_pr(cir:Circuit,index:int):
    '''
    用来清理pr 的工具函数
    '''
    ansatz_before = cir
    if index is not None:
        name_map = dict(zip(ansatz_before.ansatz_params_name,[f'ansatz{index}-{i}'for i in range(len(ansatz_before.ansatz_params_name))]))
    else:
        name_map = dict(zip(ansatz_before.ansatz_params_name,[f'ansatz{i}'for i in range(len(ansatz_before.ansatz_params_name))]))
    ansatz = change_param_name(ansatz_before,name_map)
    return ansatz

def Mindspore_ansatz_micro(Structure_p:np.array,
                     parameterized_pool:list[Circuit],
                     unparameterized_pool:list[Circuit],
                     num_layer:int=6,
                     n_qbits:int=8):
    """
    和 DQAS 文章描述的一致，生成权重线路
    更新了非参数化门的算符池引入
    Structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    """
    if Structure_p.shape[0] != num_layer:
        raise ValueError('Structure_p shape must be equal to num_layer')
    
    if Structure_p.shape[1] != len(parameterized_pool)+len(unparameterized_pool):
        raise ValueError('Structure_p shape must be equal to size of pool')

    if isinstance(Structure_p, np.ndarray):
        my_stp = ms.Tensor(Structure_p, ms.float32)
    else:
        my_stp = Structure_p
        
    prg = PRGenerator('encoder')
    nqbits = n_qbits
    encoder = Circuit()
    # encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)    
    encoder = encoder.as_encoder()             
        
    sub_ansatz = Circuit()
    #print(my_stp.shape)
    for layer_index in range(my_stp.shape[0]):
        for op_index in range(my_stp.shape[1]):
            if my_stp[layer_index,op_index] == 0:
                continue
            if op_index < len(parameterized_pool):
                before_ansatz = parameterized_pool[op_index]
                before_ansatz = change_param_name(circuit_fn=before_ansatz,name_map={before_ansatz.ansatz_params_name[0]:f'ansatz{layer_index}'})
                sub_ansatz += before_ansatz
            else:
                sub_ansatz += unparameterized_pool[op_index-len(parameterized_pool)]
    
    whole_ansatz = Circuit()
    whole_ansatz += wash_pr(apply(sub_ansatz,[0,1]),index=0)
    whole_ansatz += wash_pr(apply(sub_ansatz,[2,3]),index=1)
    whole_ansatz += wash_pr(apply(sub_ansatz,[4,5]),index=2)
    whole_ansatz += wash_pr(apply(sub_ansatz,[6,7]),index=3)
    whole_ansatz += wash_pr(apply(sub_ansatz,[0,2]),index=4)
    whole_ansatz += wash_pr(apply(sub_ansatz,[4,6]),index=5)
    whole_ansatz += wash_pr(apply(sub_ansatz,[0,4]),index=6)
    whole_ansatz =  wash_pr(whole_ansatz,index=None)
                
    finnal_ansatz = encoder.as_encoder() + whole_ansatz.as_ansatz()
    return finnal_ansatz


def vag_nnp_micro(Structure_params: np.array, Ansatz_params: np.array,paramerterized_pool:list[Circuit],unparamerterized_pool:list[Circuit],num_layer:int=6,n_qbits:int=8):
    """
    更新: 只在对应位置上更新nnp梯度
    用于计算梯度 Ansatz_params关于 loss 的梯度
    value,grad_ansatz_params = vag(训练数据,标签数据)
    """
    if isinstance(Structure_params, np.ndarray):
        mystp= ms.Tensor(Structure_params,ms.float32)
    else:
        mystp = Structure_params
    ansatz = Mindspore_ansatz_micro(Structure_p=mystp,parameterized_pool=paramerterized_pool,
                                    unparameterized_pool=unparamerterized_pool,
                                    num_layer=num_layer,n_qbits=n_qbits)
    
    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    op_index = [ops.Argmax()(i) for i in mystp]
    #print(f'op_index={op_index}')
    ansatz_pr=[]
    for each_sub in range(7):
        for index,i in enumerate(op_index):
            if i>=len(paramerterized_pool):
                continue
            #print(each_sub,index,i)
            ansatz_pr.append(Ansatz_params[each_sub,index,i])

    Mylayer = MQLayer(grad_ops,ms.Tensor(ansatz_pr,ms.float64).reshape(-1))


    def forward_fn(encode_p,y_label):
        eval_obserables = Mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss
    # nnp = ms.Tensor(Ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn,grad_position=None,weights=Mylayer.trainable_params())
    
    return grad_fn


def zeroslike_grad_nnp_micro(batch_sturcture: Union[np.ndarray, ms.Tensor], grad_nnp: Union[np.ndarray, ms.Tensor], shape_parametized: int, ansatz_parameters: np.ndarray) -> np.ndarray:
    '''
    用于根据算出的梯度更新ansatz参数    
    '''
    if isinstance(batch_sturcture, np.ndarray):
        mystp = ms.Tensor(batch_sturcture, ms.float32)
    else:
        mystp = batch_sturcture  # 如果 batch_sturcture 已经是 ms.Tensor 类型

    op_index = [ops.Argmax()(i) for i in mystp]
    #print(op_index)
    zeros_grad_nnp = np.zeros_like(ansatz_parameters)
    count = 0
    for each_sub in range(7):
        for index,i in enumerate(op_index):
            if i >= shape_parametized:
                continue
            zeros_grad_nnp[each_sub, index,i] = grad_nnp[count]
            count += 1
        
    return zeros_grad_nnp