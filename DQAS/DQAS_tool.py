import numpy as np
from mindquantum.core.gates import RX, RY, RZ, H, X, Y, Z, CNOT
from mindquantum.core.circuit import Circuit
import mindspore as ms
from mindquantum.simulator import  Simulator
from mindquantum.core.gates import GroupedPauli
from mindquantum.core.operators import TimeEvolution,QubitOperator
from mindquantum.core.parameterresolver import PRConvertible,PRGenerator,ParameterResolver
import random
from mindspore import Tensor,ops




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


def one_hot(labels, num_classes):
    one_hot_op = ops.OneHot()
    on_value = Tensor(1.0, ms.int32)
    off_value = Tensor(0.0, ms.int32)
    return one_hot_op(labels, num_classes, on_value, off_value)