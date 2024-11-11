import numpy as np
from mindquantum.core.gates import RX, RY, RZ, H, X, Y, Z, CNOT
from mindquantum.core.circuit import Circuit,UN
import mindspore as ms
from mindquantum.simulator import  Simulator
from mindquantum.core.gates import GroupedPauli
from mindquantum.core.operators import TimeEvolution,QubitOperator
from mindquantum.core.parameterresolver import PRGenerator
import random
from mindspore import Tensor,ops,Parameter
from mindquantum.core.gates import UnivMathGate
from mindquantum.core.operators import Hamiltonian             # 导入Hamiltonian模块，用于构建哈密顿量
from mindquantum.framework import MQLayer,MQOps
import mindspore.numpy as mnp
import sys
sys.path.append('..')
from Test_tool import Test_ansatz
from data_processing import X_train,X_test,y_train,y_test



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


# n_parameterized = 12
# n_layer = 6
# stddev = 0.02
# shape_nnp = (n_layer, n_parameterized)
# nnp = np.random.normal(loc=0.0, scale=stddev, size=shape_nnp).astype(np.float64)
# nnp = np.random.normal(loc=0.0, scale=stddev, size=shape_nnp).astype(np.float64)
# np.random.seed(10)
# unbound_opeartor_pool = [generate_pauli_string(n=8,seed=i)[0] for i in range(n_parameterized)]
# bound_opeartor_pool = [generate_pauli_string(n=8,seed=i)[1] for i in range(8,12)]
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


def sampling_from_structure(structures: np.array,num_layer:int,shape_parametized:int):
    softmax = ops.Softmax()
    prob = softmax(ms.Tensor(structures,ms.float32))
    #print(prob)
    prob_np = prob.asnumpy()  # 将 MindSpore Tensor 转换为 NumPy 数组
    samples = []
    for i in range(num_layer):
        sample = np.random.choice(prob_np[i].shape[0], p=prob_np[i])
        samples.append(sample)
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
    encoder += UN(H, nqbits)                                 
    for i in range(nqbits):                                  
        encoder += RY(prg.new()).on(i)                 
        
    ansatz = Circuit()
    pr_gen = PRGenerator('ansatz')
    #print(my_stp.shape)
    for i in range(num_layer):
        paramertized_part_count=0
        for index_op,each_op in enumerate(parameterized_pool):
            if my_stp[i,index_op] == 0:
                continue
            #print(my_stp[i,index_op])
            ansatz += TimeEvolution(QubitOperator(terms=each_op,coefficient=pr_gen.new()),time=float(my_stp[i,index_op])).circuit
            paramertized_part_count+=1
            
        for index_op,each_op in enumerate(unparameterized_pool):
            if my_stp[i,index_op+paramertized_part_count] == 0:
                continue
            op = GroupedPauli(each_op)
            tmp_cir = Circuit([GroupedPauli(each_op).on(range(n_qbits))])
            matrix = tmp_cir.matrix()
            ansatz += UnivMathGate(matrix_value=matrix*float(my_stp[i,index_op+paramertized_part_count]),name=f'{my_stp[i,index_op+paramertized_part_count]}*{op.pauli_string}').on(range(n_qbits))  
    
    finnal_ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    return finnal_ansatz



def vag_nnp2(Structure_params: np.array, Ansatz_params: np.array,paramerterized_pool:list,unparamerterized_pool:list,num_layer:int=6,n_qbits:int=8):
    """
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
    #print(Ansatz_params.shape)
    nnp_index = [ops.Argmax()(i) for i in ms.Tensor(mystp,ms.float32)]
    print(nnp_index)
    ansatz_parameters = [Ansatz_params[layer_index][i] for layer_index,i in enumerate(nnp_index)]
    print(ansatz_parameters)
    
    Mylayer = MQLayer(grad_ops,ms.Tensor(ansatz_parameters,ms.float64).reshape(-1))


    def forward_fn(encode_p,y_label):
        eval_obserables = Mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss
    # nnp = ms.Tensor(Ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn,grad_position=None,weights=Mylayer.trainable_params())
    
    return grad_fn