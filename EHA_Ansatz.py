import numpy as np                                          # 导入numpy库并简写为np
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ   # 导入量子门H, X, Y, Z, RX, RY, RZ
from mindquantum.core.circuit import Circuit,change_param_name
from mindquantum.simulator import Simulator

class EHA():
    """
    Entanglement-variational Hardware-efficient Ansatz (EHA)
    """
    def __init__(self,n_qubit:int,Layer:int) -> None:
        self.EHA_ansatz = Circuit()
        self.n_qubit = n_qubit
        self.Layer = Layer
        self.simulator = Simulator('mqvector',self.n_qubit)
        for i in range(Layer):
            rotation_block =  self.rotation_part()
            rename_rb = change_param_name(circuit_fn=rotation_block,name_map=dict(zip(rotation_block.params_name,['Layer'+str(i)+'_theta'+str(j) for j in range(len(rotation_block.params_name))])))
            self.EHA_ansatz+=rename_rb
            
            for j in range(self.n_qubit)[:self.n_qubit-1]:
                tmp= self.Entangle_part(former_qubit=j,latter_qubit=j+1)
                entangle_block = change_param_name(circuit_fn=tmp,name_map=dict(zip(tmp.params_name,['Layer'+str(i)+'_'+str(j)+'_Beta'+str(k) for k in range(3)])))         
                self.EHA_ansatz+=entangle_block        
        
        
        
    def rotation_part(self):
        rotation_part = Circuit()
        for index,i in enumerate(range(self.n_qubit)):
            rotation_part +=RX(f'theta_{index}').on(i)
        
        return rotation_part
            
        
    def Entangle_part(self,former_qubit:int,latter_qubit:int):
        Entangle_part = Circuit()
        #XX
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX('Beta_1').on(obj_qubits=former_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        #YY
        Entangle_part += RX(np.pi/2).on(obj_qubits=former_qubit)
        Entangle_part += RX(np.pi/2).on(obj_qubits=latter_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX('Beta_2').on(obj_qubits=latter_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX(-1*np.pi/2).on(obj_qubits=former_qubit)
        Entangle_part += RX(-1*np.pi/2).on(obj_qubits=latter_qubit)
        #ZZ
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        Entangle_part += RX('Beta_3').on(obj_qubits=latter_qubit)
        Entangle_part += X.on(obj_qubits=latter_qubit,ctrl_qubits=former_qubit)
        return Entangle_part
        