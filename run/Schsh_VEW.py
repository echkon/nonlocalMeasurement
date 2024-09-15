import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, sigmax, sigmaz, ket2dm
from base.entanglementMeasure import *
from base.witness import variationalWitness


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

# Run program
thetas = np.linspace(0, 1, 100)
phis = [0, np.pi/4, np.pi/2, np.pi]

Schsh = [[], [], [], []]
witness = [[], [], [], []]

for i, phi in enumerate(phis):    
    for theta in thetas:
        theta = theta * np.pi
        state = psi(theta, phi)

        EXXtheo = np.real((state.dag() * XX * state))
        EZZtheo = np.real((state.dag() * ZZ * state))
        Schsh[i].append(- 2/np.sqrt(2) * (EXXtheo + EZZtheo))
        
        omg = [4.0, 0.0]  # Initial guess for optimization
        vew = variationalWitness(omg, state).fit()
        witness[i].append(np.real(vew))

plotPPT = []
plotConc = []

theta_theory = np.linspace(0, 1, 200)
for theta in theta_theory: 
    theta = theta * np.pi
    quantum_state = ket2dm(psi(theta, 0))
    plotPPT.append(np.real(ppt_criterion(quantum_state)))
    plotConc.append(np.real(concurrence_value(quantum_state)))

# Plot figure        
fig, ax = plt.subplots(2, figsize=(8, 12))

ax[0].fill_between(thetas, -2, -3, alpha=0.2)
ax[0].plot(theta_theory, plotPPT, label="ppt_phi=0")
ax[0].plot(theta_theory, plotConc, label="concurrence") 
ax[0].plot(thetas, Schsh[0], linestyle='--', label='phi=0')
ax[0].plot(thetas, Schsh[1], linestyle='--', label='phi=pi/4')
ax[0].plot(thetas, Schsh[2], linestyle='--', label='phi=pi/2')
ax[0].plot(thetas, Schsh[3], linestyle='--', label='phi=pi')

ax[1].plot(theta_theory, plotPPT, label="ppt_phi=0")
ax[1].plot(theta_theory, plotConc, label="concurrence") 
ax[1].plot(thetas, witness[0], linestyle='--', label='Wphi=0')
ax[1].plot(thetas, witness[1], linestyle='--', label='Wphi=pi/4')
ax[1].plot(thetas, witness[2], linestyle='--', label='Wphi=pi/2')
ax[1].plot(thetas, witness[3], linestyle='--', label='Wphi=pi')

plt.legend()
plt.show()
plt.savefig('Schsh_n_VEW.eps')
plt.savefig('Schsh_n_VEW.png')