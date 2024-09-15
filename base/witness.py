import numpy as np
from scipy.optimize import minimize
from qutip import basis, tensor, sigmax, sigmaz, ket2dm

# Define bases
up = basis(2, 0)
dn = basis(2, 1)
X = sigmax()
Z = sigmaz()
XX = tensor(X, X)
ZZ = tensor(Z, Z)

class variationalWitness():
    def __init__(self, theta, input_state, sep_state=None, penalty=10):
        self.theta = theta  # initial values for theta
        self.input_state = ket2dm(input_state)  # input state
        self.sep_state = sep_state  # separable states
        self.penalty = penalty  # penalty

    # Define the parameterized entanglement witness W(theta)
    def entanglement_witness(self):
        W = self.theta[0] * ZZ + self.theta[1] * XX
        return -np.sqrt(2.0) * W

    def sep_condition(self):
        if self.sep_state is not None:
            return self.sep_state
        else:
            # Define some separable states
            rho_sep1 = tensor(up, up) * tensor(up, up).dag()  # |00⟩ state
            rho_sep2 = tensor(up, dn) * tensor(up, dn).dag()  # |01⟩ state
            rho_sep3 = tensor(dn, up) * tensor(dn, up).dag()  # |10⟩ state
            rho_sep4 = tensor(dn, dn) * tensor(dn, dn).dag()  # |11⟩ state
            separable_states = [rho_sep1, rho_sep2, rho_sep3, rho_sep4]
            return separable_states

    # Cost function with penalty
    def cost_function(self, theta):
        # Update theta values
        self.theta = theta
        sep_state = self.sep_condition()
        penalty = self.penalty
        rho = self.input_state
        W = self.entanglement_witness()
        
        # Expectation value for the input (entangled) state
        expectation_entangled = (W * rho).tr()
        
        # Apply penalty for separable states
        sum_penalty = sum(max(0, -np.real((W * sigma).tr())) for sigma in sep_state)
        
        return np.real(expectation_entangled) + penalty * sum_penalty

    def fit(self):
        theta = self.theta
        
        # Minimize the cost function
        result = minimize(self.cost_function, theta, method='COBYLA')
        self.theta = result.x
        exp_value = (self.entanglement_witness() * self.input_state).tr()
        #print("Optimal theta:", self.theta)
        return np.real(exp_value)
