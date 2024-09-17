import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tqix as tq

# Pauli matrices and tensor products
X, Y, Z = tq.sigmax(), tq.sigmay(), tq.sigmaz()
XX, YY, ZZ = tq.tensorx(X, X), tq.tensorx(Y, Y), tq.tensorx(Z, Z)

# Werner state function
def werner(p):
    psi_minus = np.array([0, 1, -1, 0]) / np.sqrt(2)
    rho_psi_minus = np.outer(psi_minus, np.conj(psi_minus))
    return p * rho_psi_minus + (1 - p) * np.eye(4, dtype=complex) / 4

# Generalized entanglement witness function
def entanglement_witness(omg, extended=False):
    W = omg[0] * XX + omg[1] * ZZ
    if extended:
        W += omg[2] * YY
        return W
    else:
        return -np.sqrt(2.0)*W

# Cost function with penalty
def cost_function(omg, rho, sep_states, lambda_penalty=10, extended=False):
    W = entanglement_witness(omg, extended)
    expectation = np.trace(W @ rho)
    penalty = sum(max(0, -np.trace(W @ sigma)) for sigma in sep_states)
    return np.real(expectation) + lambda_penalty * penalty

# Separable states
rho_sep = [np.outer([1, 0, 0, 0], [1, 0, 0, 0]), np.outer([0, 1, 0, 0], [0, 1, 0, 0]),
           np.outer([0, 0, 1, 0], [0, 0, 1, 0]), np.outer([0, 0, 0, 1], [0, 0, 0, 1]),
           werner(0.0)]

# Plotting function for entanglement witness
def plot_witness(ps, extended=False):
    witness_values = []
    for p in ps:
        ave = 0
        for _ in range(10 if not extended else 1):
            initial_guess = [0, 4 * np.random.random(), -4 * np.random.random()] if extended else [0, 4 * np.random.random()]
            rho = werner(p)
            result = minimize(cost_function, initial_guess, args=(rho, rho_sep, 10, extended), method='COBYLA')
            ave += np.trace(entanglement_witness(result.x, extended) @ rho)
        witness_values.append(ave / (10 if not extended else 1))
    return witness_values

# Plot results
ps, ps_3 = np.linspace(0, 1, 500), np.linspace(0, 1, 100)
plt.plot(ps, plot_witness(ps), linewidth=2, label="Witness")
plt.plot(ps_3, plot_witness(ps_3, extended=True), linestyle='--', label="Witness_3")
plt.axvline(x=1/3, color='r', linestyle='--')
plt.plot(ps, [0] * ps)
plt.legend()
plt.savefig("witnessW_new.eps")
plt.savefig("witnessW_new.png")
