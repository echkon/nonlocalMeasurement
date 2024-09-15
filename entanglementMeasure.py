import numpy as np
from scipy.linalg import logm  
from qutip import partial_transpose, concurrence

def ppt_criterion(rho):
    # [0,1] first '0' is the subsystem. '1' is the qubit to be transposed.
    rho_pt = partial_transpose(rho, [0,1])
    rho_pt_eig = np.linalg.eigvals(rho_pt.full()) 
    return np.min(rho_pt_eig)

def concurrence_value(rho):
    concurrence_value = concurrence(rho)
    return concurrence_value