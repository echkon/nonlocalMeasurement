import sys
sys.path.insert(0, '../../../')
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sigmax, sigmaz, ket2dm
from base.entanglementMeasure import *
from base.witness import variationalWitness
from base.randomState import pureStates
import pickle

#define bases
up = basis(2, 0)
dn = basis(2, 1)
X = sigmax()
Z = sigmaz()
XX = tensor(X, X)
ZZ = tensor(Z, Z)

# generate states
states = pureStates()

# or load from the file
#with open("data4C_random.pkl", "rb") as f:
#    states, VEW = pickle.load(f)

Schsh = []
witness = []

#sep
sep_states = []
for state in states[0:4]:
    sep_states.append(ket2dm(state)) 

# run    
for state in states:
    EXXtheo = np.real((state.dag() * XX * state))[0,0]
    EZZtheo = np.real((state.dag() * ZZ * state))[0,0]
    Schsh.append(- 2/np.sqrt(2) * (EXXtheo + EZZtheo))
        
    omg = [4,0]#[np.random.rand(), np.random.rand()]  # Initial guess for optimization
    vew = variationalWitness(omg, state, sep_state = sep_states).fit()
    #print(vew)
    witness.append(np.real(vew))

plotPPT = []
plotConc = []

for state in states: 
    quantum_state = ket2dm(state)
    plotPPT.append(np.real(ppt_criterion(quantum_state)))
    plotConc.append(np.real(concurrence_value(quantum_state)))


# Save to a file
with open("data4C_random.pkl", "wb") as f:
    pickle.dump((states, witness), f)

# Plot figure        
fig, ax = plt.subplots(2, figsize=(8, 12))
idx = list(range(len(states)))

ax[0].fill_between(idx, -2, -3, alpha=0.2)
ax[0].plot(idx, plotPPT, label="ppt")
ax[0].plot(idx, plotConc, label="concurrence") 
ax[0].plot(idx, Schsh, linestyle='-.', label='CHSH')
ax[1].plot(idx, np.array(witness), linestyle='--', label='Witness')

ax[1].plot(idx, plotPPT, label="ppt")
ax[1].plot(idx, plotConc, label="concurrence") 

plt.legend()
plt.show()
plt.savefig('pure_VEW.eps')
plt.savefig('pure_VEW.png')