from qiskit import QuantumCircuit
import numpy as np
import qiskit.quantum_info as qi
from qiskit.primitives import Sampler

class nonlocalCircuit: 
    def __init__(self, theta, phi): 
        self.theta = theta 
        self.phi = phi 
           
    def circuit(self): 
        # creat a quantum circuit with 6 qubits, 4 clbits
        qc = QuantumCircuit(6,4) 
        qc.u(2*self.theta,self.phi,0,0)
        qc.cx(0,1)
        
        qc.h(2)
        qc.cx(2,3) 
        
        qc.cx(0,2) 
        qc.cx(1,3) 
                
        qc.h(4)
        qc.x(5)

        qc.cx(4,5)
        
        qc.cx(4,0)
        qc.cx(5,1)
        
        qc.h(4)
        qc.h(5)
        
        qc.measure([2,3],[0,1]) 
        qc.measure([4,5],[2,3])

        return qc 
    
    def probability(self): 
        qc = self.circuit()
        # Initialize Sampler and run it
        sampler = Sampler()
        result = sampler.run(qc, shots=10000).result().quasi_dists[0]   
        return result 
    
    def bitString_probability(self):
        prob = self.probability() 
        prob = {'{:04b}'.format(key): value for key, value in prob.items()}
        str4bits  = [format(i, '04b') for i in range(16)]
        for i in str4bits: 
            if i not in prob: 
                prob[i] = 0 
        return prob 
   
    def expectation(self):
        prob = self.probability() 
        prob = {'{:04b}'.format(key): value for key, value in prob.items()}
        str4bits  = [format(i, '04b') for i in range(16)]

        for i in str4bits: 
            if i not in prob: 
                prob[i] = 0 

        p00x = prob['0000'] + prob['0001'] + prob['0010'] + prob['0011']
        p01x = prob['0100'] + prob['0101'] + prob['0110'] + prob['0111']
        p10x = prob['1000'] + prob['1001'] + prob['1010'] + prob['1011']
        p11x = prob['1100'] + prob['1101'] + prob['1110'] + prob['1111']
        
        probX = p00x - p01x -p10x + p11x 
        
        p00z = prob['0000'] + prob['0100'] + prob['1000'] + prob['1100']
        p01z = prob['0001'] + prob['0101'] + prob['1001'] + prob['1101']    
        p10z = prob['0010'] + prob['0110'] + prob['1010'] + prob['1110']
        p11z = prob['0011'] + prob['0111'] + prob['1011'] + prob['1111']
        probZ = p00z - p01z - p10z + p11z
        
        return  probZ, probX 

    def state(self): 
        qc = self.circuit()
                
        # Remove measurement instructions
        qc_no_measurements = qc.copy()
        qc_no_measurements.remove_final_measurements()
        
        density_matrix = qi.DensityMatrix.from_instruction(qc_no_measurements)
        reduced_state = qi.partial_trace(density_matrix, [2, 3, 4, 5])
    
        return reduced_state
    
    def post_state(self):
        qc = self.circuit() 
        pstate = qi.DensityMatrix(qc)
        return pstate


