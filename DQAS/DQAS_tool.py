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


n_parameterized = 12
n_layer = 6
stddev = 0.02
shape_nnp = (n_layer, n_parameterized)
nnp = np.random.normal(loc=0.0, scale=stddev, size=shape_nnp).astype(np.float64)
# nnp = np.random.normal(loc=0.0, scale=stddev, size=shape_nnp).astype(np.float64)
unbound_opeartor_pool = [generate_pauli_string(n=8,seed=i)[0] for i in range(n_parameterized)]
bound_opeartor_pool = [generate_pauli_string(n=8,seed=i)[1] for i in range(8,12)]
loss_fn = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean') 

def one_hot(labels, num_classes):
    one_hot_op = ops.OneHot()
    on_value = Tensor(1.0, ms.int32)
    off_value = Tensor(0.0, ms.int32)
    return one_hot_op(labels, num_classes, on_value, off_value)




def Mindspore_ansatz(Structure_p:np.array,n_layer:int,n_qbits:int=8):
    """
    和 DQAS 文章描述的一致，生成权重线路
    Structure_p:np.array DQAS中的权重参数,
    Ansatz_p:np.array  DQAS中的Ansatz参数,
    
    """
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
    for i in range(n_layer):
        paramertized_part_count=0
        for index_op,each_op in enumerate(unbound_opeartor_pool):
            ansatz += TimeEvolution(QubitOperator(terms=each_op,coefficient=pr_gen.new()),time=float(my_stp[i,index_op])).circuit
            paramertized_part_count+=1
            
        # for index_op,each_op in enumerate(bound_opeartor_pool):
        #     op = GroupedPauli(each_op)
        #     tmp_cir = Circuit([GroupedPauli(each_op).on(range(n_qbits))])
        #     matrix = tmp_cir.matrix()
        #     ansatz += UnivMathGate(matrix_value=matrix*float(my_stp[i,index_op+paramertized_part_count]),name=f'{my_stp[i,index_op+paramertized_part_count]}*{op.pauli_string}').on(range(n_qbits))  
    
    finnal_ansatz = encoder.as_encoder() + ansatz.as_ansatz()
    return finnal_ansatz
    



def vag_nnp(Structure_params: np.array, Ansatz_params: np.array,n_layer:int=3,n_qbits:int=8):
    """
    用于计算梯度 Ansatz_params关于 loss 的梯度
    value,grad_ansatz_params = vag(训练数据,标签数据)
    """
    ansatz = Mindspore_ansatz(Structure_params, n_layer=n_layer, n_qbits=n_qbits)
    sim = Simulator(backend='mqvector', n_qubits=n_qbits)
    hams = [Hamiltonian(QubitOperator(f'Z{i}')) for i in [0, 1]]
    grad_ops = sim.get_expectation_with_grad(hams, ansatz)
    print(Ansatz_params.shape)
    Mylayer = MQLayer(grad_ops,ms.Tensor(Ansatz_params,ms.float64).reshape(-1))


    def forward_fn(encode_p,y_label):
        eval_obserables = Mylayer(encode_p)
        loss = loss_fn(eval_obserables, y_label)
        return loss
    # nnp = ms.Tensor(Ansatz_params).reshape(-1)
    grad_fn = ms.value_and_grad(fn=forward_fn,grad_position=None,weights=Mylayer.trainable_params())
    
    return grad_fn


def vag_nnp_function(Structure_params,Ansatz_params,X_train, y_train):
    grad_fn = vag_nnp(Structure_params, Ansatz_params, n_layer=3, n_qbits=8)
    loss, grads = grad_fn(ms.Tensor(X_train), ms.Tensor(y_train))
    return loss, grads


def sampling_from_structure(structures: np.array,num_layer:int,shape_parametized:int):
    softmax = ops.Softmax()
    prob = softmax(ms.Tensor(structures,ms.float32))
    #print(prob)
    prob_np = prob.asnumpy()  # 将 MindSpore Tensor 转换为 NumPy 数组
    samples = []
    for i in range(num_layer):
        sample = np.random.choice(shape_parametized, p=prob_np[i])
        samples.append(sample)
    return np.array(samples)
