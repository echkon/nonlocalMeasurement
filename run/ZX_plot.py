import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sigmax, sigmaz
from base.quantumCircuits import nonlocalCircuit

#define bases
up = basis(2, 0)
dn = basis(2, 1)
X = sigmax()
Z = sigmaz()
XX = tensor(X, X)
ZZ = tensor(Z, Z)

#define quantum state |psi>
def psi(theta, phi):
    state = np.cos(theta) * tensor(up, up) + np.exp(1j * phi) * np.sin(theta) * tensor(dn, dn)
    return state

thetas = np.linspace(0, 1, 100)
phis = [0, np.pi/4, np.pi/2, np.pi]

XXTheo = [[], [], [], []] 
XXPovm = [[], [], [], []]
ZZTheo = [[], [], [], []] 
ZZPovm = [[], [], [], []]

errors = [[], [], [], []]

for i, phi in enumerate(phis):    
    for theta in thetas:
        theta = theta * np.pi
        state = psi(theta, phi)
        
        XXTheo[i].append(np.real((state.dag() * XX * state)))
        ZZTheo[i].append(np.real((state.dag() * ZZ * state)))
        
        qc = nonlocalCircuit(theta, phi)
        EZZPovm = qc.expectation()[0]
        EXXPovm = qc.expectation()[1]
        
        XXPovm[i].append(EXXPovm)
        ZZPovm[i].append(EZZPovm)
                
error = (np.array(XXTheo) - np.array(XXPovm)) ** 2
    
fig, ax = plt.subplots(2, figsize = (8,12))
ax[0].plot(thetas, XXTheo[0], linestyle = '--', label='phi=0')
ax[0].plot(thetas, XXTheo[1], linestyle = '--', label='phi=pi/4')
ax[0].plot(thetas, XXTheo[2], linestyle = '--', label='phi=pi/2')
ax[0].plot(thetas, XXTheo[3], linestyle = '--', label='phi=pi')

ax[0].plot(thetas, XXPovm[0], label='phi=0')
ax[0].plot(thetas, XXPovm[1], label='phi=pi/4')
ax[0].plot(thetas, XXPovm[2], label='phi=pi/2')
ax[0].plot(thetas, XXPovm[3], label='phi=pi')

ax[0].plot(thetas, ZZTheo[0], linestyle = '--', label='phi=0')
ax[0].plot(thetas, ZZTheo[1], linestyle = '--', label='phi=pi/4')
ax[0].plot(thetas, ZZTheo[2], linestyle = '--', label='phi=pi/2')
ax[0].plot(thetas, ZZTheo[3], linestyle = '--', label='phi=pi')

ax[0].plot(thetas, ZZPovm[0], label='phi=0')
ax[0].plot(thetas, ZZPovm[1], label='phi=pi/4')
ax[0].plot(thetas, ZZPovm[2], label='phi=pi/2')
ax[0].plot(thetas, ZZPovm[3], label='phi=pi')

ax[1].set_yscale('log')

ax[1].plot(thetas, error[0], label = 'phi=0')
ax[1].plot(thetas, error[1], label = 'phi=pi/4')
ax[1].plot(thetas, error[2], label = 'phi=pi/2')
ax[1].plot(thetas, error[3], label = 'phi=pi')


plt.legend()
plt.show()
plt.savefig('ZX_plot.eps')
plt.savefig('ZX_plot.png')    