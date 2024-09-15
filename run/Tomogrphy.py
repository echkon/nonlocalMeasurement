
import numpy as np
import matplotlib.pyplot as plt
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.visualization import plot_state_city
from base.quantumCircuits import nonlocalCircuit


qc = nonlocalCircuit(np.pi/4, np.pi/4)
tomo_circuits = state_tomography_circuits(qc.circuit(), [0, 1])
result = qc.probability()

# Step 7: Fit the tomography results
tomo_fitter = StateTomographyFitter(result, tomo_circuits)
rho_fit = tomo_fitter.fit(method='lstsq')

# Step 8: Visualize the fitted density matrix
plot_state_city(rho_fit)
plt.savefig('Tomography.png')













