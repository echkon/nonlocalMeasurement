import sys
sys.path.insert(0, '../')
from qutip import qeye 
import matplotlib.pyplot as plt 
from qutip import basis, tensor, ket2dm, sigmax, sigmaz
from base.entanglementMeasure import *

#define bases
up = basis(2, 0)
dn = basis(2, 1)
X = sigmax()
Z = sigmaz()
XX = tensor(X, X)
ZZ = tensor(Z, Z)

def bell_state():
    state = 1/np.sqrt(2) * (tensor(up, dn) - tensor(dn, up))
    return ket2dm(state)

def werner_state(w):
    IAB = tensor(qeye(2), qeye(2))
    bstate = bell_state() 
    w_state = w * bstate + (1 - w) * IAB / 4 
    return w_state 

def ZZpovm(a): 
    rho = werner_state(a)
    ZZPOVM = (rho * ZZ).tr()
    return ZZPOVM

def XXpovm(a): 
    rho = werner_state(a)
    XXPOVM = (rho * XX).tr()
    return XXPOVM

def Schsh(a): 
    rho = werner_state(a)
    ZZPOVM = (rho * ZZ).tr()
    XXPOVM = (rho * XX).tr()
    SCHSH_POVM = -np.sqrt(2) * (ZZPOVM + XXPOVM)
    return SCHSH_POVM

ppt_ws = []
conc_ws = []
Schshwx = []
ZZpovmData = [] 
XXpovmData = [] 
SchshPovmData = [] 
wx = np.linspace(0, 1, 10)  

for i in wx: 
    quantum_state = werner_state(i)
    ppt_ws.append(ppt_criterion(quantum_state))
    conc_ws.append(concurrence_value(quantum_state))
    Schshwx.append(Schsh(i))
    ZZpovmData.append(ZZpovm(i))
    XXpovmData.append(XXpovm(i))
    SchshPovmData.append(Schsh(i))

plt.plot(wx, ppt_ws , label = "PPT") 
plt.plot(wx, conc_ws, label = "concurrence") 
plt.plot(wx, Schshwx, label = "Mixed State Epectation")
plt.plot(wx, len(wx) * [2], label = "Threshold of entanglemet")
plt.plot(wx, SchshPovmData, label = "Schsh Povm")

plt.legend()
plt.savefig("Mixed_State.png")
plt.savefig("Mixed_State.eps")