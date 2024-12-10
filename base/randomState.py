import numpy as np
from qutip import basis, tensor, sigmax, sigmaz, ket2dm

def pureStates():
    #create 100 pure state
    #|psi> = a|00> + b|01> + c|10> + d|11>
    
    up = basis(2, 0)
    dn = basis(2, 1)
    states = []
    
    for i in range(50):
        a, b, c, d = generate_separable_state()
        st = a*tensor(up,up) + b*tensor(up,dn) + c*tensor(dn,up) + d*tensor(dn,dn)
        states.append(st)
    for i in range(50):
        a, b, c, d = generate_ad_cossine() #calculate_abcd_rand() #generate_entangled_state()
        st = a*tensor(up,up) + b*tensor(up,dn) + c*tensor(dn,up) + d*tensor(dn,dn)
        states.append(st)

    return states    

# Function to generate separable state (C = 0)
def generate_separable_state():
    # Choose random a, b, c, and compute d such that ad = bc
    a = np.random.rand() #+ 1j * np.random.rand()  # random complex number
    b = np.random.rand() #+ 1j * np.random.rand()
    c = np.random.rand() #+ 1j * np.random.rand()
    d = b * c / a if a != 0 else 0  # Ensures ad = bc
    
    # Normalize the state
    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2)
    a /= norm
    b /= norm
    c /= norm
    d /= norm

    return a, b, c, d

# Function to generate entangled state (C > 0)
def generate_entangled_state():
    while True:
        a = np.random.rand() #+ 1j * np.random.rand()
        b = np.random.rand() #+ 1j * np.random.rand()
        c = np.random.rand() #+ 1j * np.random.rand()
        d = np.random.rand() #+ 1j * np.random.rand()
        
        # Calculate concurrence C = 2|ad - bc|
        # C = 2 * np.abs(a * d - b * c)
        C = 1 #set so
        
        if C > 0:  # Check if the state is entangled
            # Normalize the state
            norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2)
            a /= norm
            b /= norm
            c /= norm
            d /= norm
            return a, b, c, d

def calculate_abcd():
    # Choose free variables a and b in range [-1, 1]
    
    while True:
        # Step 1: Randomly generate a, b such that a^2 + b^2 <= 1
        a, b = np.random.uniform(-1, 1, size=2)
        if a**2 + b**2 > 1:
            continue

        # Step 2: Set R = sqrt(1 - (a^2 + b^2))
        R = np.sqrt(1 - (a**2 + b**2))

        # Step 3: Solve for c, d to satisfy 2|ad - bc| = 1
        # Assume c = R * cos(theta) and d = R * sin(theta)
        # Use |ad - bc| = 1/2 to find theta
        target = 1 / (2 * R)
        if target > 1:
            continue  # Skip invalid cases where target exceeds 1

        theta = np.arcsin(target)  # Solve for theta
        c = R * np.cos(theta)
        d = R * np.sin(theta)

        # Verify if C = 1
        C = 2 * abs(a * d - b * c)
        if np.isclose(C, 1, atol=1e-6):
            return a, b, c, d    


def calculate_abcd_rand():
    # Choose free variables a and b in range [-1, 1]
    
    def random_ab():
        while True:
            # Generate random numbers in [-1, 1]
            a = np.random.uniform(-1, 1)
            b = np.random.uniform(-1, 1)
            # Check the condition a^2 + b^2 <= 1
            if a**2 + b**2 <= 1:
                R = np.sqrt(1 - (a**2 + b**2))
                if R > 0.5:
                    return a, b
        
    def solve_cd_direct(a, b):
        # Ensure a^2 + b^2 <= 1
        norm_ab2 = a**2 + b**2
        if norm_ab2 >= 1:
            raise ValueError("a^2 + b^2 must be less than 1 for valid solutions.")

        # Calculate R
        R = np.sqrt(1 - norm_ab2)
        
        # Solve for theta
        target = 1 / (2 * R)
        if target > 1:
            raise ValueError("No solution exists for given a and b.")

        # Solve a*sin(theta) - b*cos(theta) = ±target
        # This can be rewritten as tan(theta) = (a ± target) / b
        theta1 = np.arctan2(target, b)  # One solution
        theta2 = np.arctan2(-target, b)  # Another solution

        # Compute c and d for each theta
        c1, d1 = R * np.cos(theta1), R * np.sin(theta1)
        c2, d2 = R * np.cos(theta2), R * np.sin(theta2)

        return c1, d1  #(c2, d2)
    a, b = random_ab() 
    c, d = solve_cd_direct(a,b)
    return a,b,c,d


def generate_ad_cossine():
    
    theta = np.random.uniform(0, np.pi / 2)
    phi = np.random.uniform(0, 2 * np.pi)
    
    a = np.cos(theta)
    d = np.sin(theta) * np.exp(1j * phi)
    b = 10e-12
    c = 10e-12
    
    # Verify the condition a^2 + |b|^2 = 1
    #assert np.isclose(a**2 + abs(d)**2, 1), "Condition a^2 + |b|^2 = 1 is not satisfied"
    norm = np.sqrt(np.abs(a)**2 + np.abs(b)**2 + np.abs(c)**2 + np.abs(d)**2)
    a /= norm
    b /= norm
    c /= norm
    d /= norm
    
    return a, b, c, d

import numpy as np
from qutip import ket, tensor

def bell_diagonal(num_states=100):
    """
    Generate Bell diagonal states with 50 separable and 50 entangled states using QuTiP.

    Args:
        num_states (int): Total number of Bell diagonal states to generate (default 100).

    Returns:
        tuple: Two lists of density matrices:
            - separable_states: List of 50 separable states.
            - entangled_states: List of 50 entangled states.
    """
    if num_states % 2 != 0:
        raise ValueError("The number of states must be even to ensure 50% separable and 50% entangled.")

    # Define the four Bell states
    bell_states = [
        tensor(ket("00") + ket("11")).unit(),
        tensor(ket("00") - ket("11")).unit(),
        tensor(ket("01") + ket("10")).unit(),
        tensor(ket("01") - ket("10")).unit()
    ]

    def create_bell_diagonal_state(coefficients):
        """Create a Bell diagonal state as a density matrix."""
        rho = sum(c * bs * bs.dag() for c, bs in zip(coefficients, bell_states))
        return rho

    separable_states = []
    entangled_states = []

    while len(separable_states) < num_states // 2 or len(entangled_states) < num_states // 2:
        # Generate random eigenvalues summing to 1
        eigenvalues = np.random.dirichlet(np.ones(4), size=1)[0]
        max_lambda = max(eigenvalues)

        # Check if the state is separable or entangled
        state = create_bell_diagonal_state(eigenvalues)
        if max_lambda <= 0.5 and len(separable_states) < num_states // 2:
            separable_states.append(state)
        elif max_lambda > 0.5 and len(entangled_states) < num_states // 2:
            entangled_states.append(state)

    return separable_states + entangled_states


def rho_high_dim(d):
    rhos = []
    
    for i in range(d):
        psi = tensor(basis(d, i), basis(d, i))
        rhos.append(ket2dm(psi))
    
    psi = sum(tensor(basis(d, i), basis(d, i)) for i in range(d)) / np.sqrt(d)
    rhos.append(ket2dm(psi))
    
    psi = sum(tensor(basis(d, i), basis(d, i)) for i in [0,d-1]) / np.sqrt(2)
    rhos.append(ket2dm(psi))
    
    return rhos
