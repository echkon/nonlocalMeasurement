import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from base.entanglementMeasure import *

from qutip import basis, tensor, ket2dm, Qobj

def create_final_state(theta, phi):
    # Step 1: Create a 4x4 zero matrix in NumPy
    final_state_np = np.zeros((4, 4), dtype=complex)  # Use dtype=complex to support complex numbers

    # Step 2: Modify the diagonal and off-diagonal elements
    final_state_np[1, 1] = 0.5
    final_state_np[2, 2] = 0.5
    final_state_np[1, 2] = 0.5 * np.cos(phi) * np.sin(2 * theta)
    final_state_np[2, 1] = 0.5 * np.cos(phi) * np.sin(2 * theta)

    # Step 3: Convert the NumPy array back to a QuTiP Qobj
    final_state = Qobj(final_state_np, dims=[[2, 2], [2, 2]])

    return final_state

thetas =  np.linspace(0, 2*np.pi, 400)
phis = [0 , np.pi/4, np.pi/2, np.pi]

PPT = [[],[],[],[]]
Conc = [[],[],[],[]]


for i, phi in enumerate(phis):    
    for theta in thetas:        
        quantumState = create_final_state(theta, phi)
        PPT[i].append(ppt_criterion(quantumState))
        Conc[i].append(concurrence_value(quantumState))


ax = plt.subplot(111, projection="polar")
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

plt.plot(thetas, PPT[0], label = "PPT phi = 0")
plt.plot(thetas, PPT[1], label = "PPT phi = pi/4")
plt.plot(thetas, PPT[2], linestyle = '--', label = "phi = pi/2")

plt.plot(thetas, Conc[0], label = "Conc phi = 0")
plt.plot(thetas, Conc[1], label ="Conc phi = pi/4")

plt.legend()
ax.set_title("A line plot on a polar axis", va='bottom')
plt.savefig("PolarPlotPPTFinalState.png")
plt.savefig("PolarPlotPPTFinalState.eps")

plt.show()