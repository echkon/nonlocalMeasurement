import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import qiskit.visualization as qiskit_vis
from base.quantumCircuits import nonlocalCircuit


theta = 0.0
phi = 0.0
qc = nonlocalCircuit(theta, phi)

plt = qiskit_vis.plot_histogram([qc.bitString_probability()], figsize = (8,8))
plt.savefig(f'Histogram{theta,phi}.png')
plt.savefig(f'Histogram{theta,phi}.eps')