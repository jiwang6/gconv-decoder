# The basis to simulate the toric code: generate grids, simulate errors, check if correction is complete etc

from random import random, seed, randint
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import hstack, kron, eye, csc_matrix, block_diag
import time


def repetition_code(n):
    """
    Parity check matrix of a repetition code with length n.
    """
    row_ind, col_ind = zip(*((i, j) for i in range(n) for j in (i, (i+1)%n)))
    data = np.ones(2*n, dtype=np.uint8)
    return csc_matrix((data, (row_ind, col_ind)))


def toric_code_x_stabilisers(L):
    """
    Sparse check matrix for the X stabilisers of a toric code with 
    lattice size L, constructed as the hypergraph product of 
    two repetition codes.
    """
    Hr = repetition_code(L)
    H = hstack(
            [kron(Hr, eye(Hr.shape[1])), kron(eye(Hr.shape[0]), Hr.T)],
            dtype=np.uint8
        )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)

def toric_code_z_stabilisers(L):
    """
    Sparse check matrix for the Z stabilisers of a toric code with 
    lattice size L, constructed as the hypergraph product of 
    two repetition codes.
    """
    Hr = repetition_code(L)
    H = hstack(
            [kron(eye(Hr.shape[0]), Hr), kron(Hr.T, eye(Hr.shape[1]))],
            dtype=np.uint8
        )
    H.data = H.data % 2
    H.eliminate_zeros()
    return csc_matrix(H)

def toric_code_x_logicals(L):
    """
    Sparse binary matrix with each row corresponding to an X logical operator 
    of a toric code with lattice size L. Constructed from the 
    homology groups of the repetition codes using the Kunneth 
    theorem.
    """
    H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    x_logicals = block_diag([kron(H1, H0), kron(H0, H1)])
    x_logicals.data = x_logicals.data % 2
    x_logicals.eliminate_zeros()
    return csc_matrix(x_logicals)

def toric_code_z_logicals(L):
    """
    Sparse binary matrix with each row corresponding to a Z logical operator 
    of a toric code with lattice size L. Constructed from the 
    homology groups of the repetition codes using the Kunneth 
    theorem.
    """
    H1 = csc_matrix(([1], ([0],[0])), shape=(1,L), dtype=np.uint8)
    H0 = csc_matrix(np.ones((1, L), dtype=np.uint8))
    z_logicals = block_diag([kron(H0, H1), kron(H1, H0)])
    z_logicals.data = z_logicals.data % 2
    z_logicals.eliminate_zeros()
    return csc_matrix(z_logicals)

def arrange_x_syndrome(syndrome):
    L = int(np.sqrt(len(syndrome)))
    syndrome = syndrome.reshape((L, L))
    # bring last row to the top
    syndrome = np.roll(syndrome, 1, axis=0)
    syndrome = syndrome.flatten()
    return syndrome

def arrange_physical_qubits(qubits):
    # arrange into 2D array
    # sqrt length of qubits
    L = int(np.sqrt(len(qubits)/2))
    qubits = qubits.reshape((2*L, L))

    # split in two
    half = int(len(qubits)/2)
    left = qubits[:half]
    right = qubits[half:]

    # zip together
    qubits = np.array(list(zip(left, right)))
    # reshape into 2D array
    qubits = qubits.reshape(2*L, L)

    #move last row to the top
    qubits = np.roll(qubits, 1, axis=0)

    #flatten
    qubits = qubits.flatten()
    return qubits

def generate_single_observation(useed):
    # generate random integer:
    seed(useed)
    print(f"seed: {useed}")
    L = 5
    percent_error = 0.050 # typical x error rate is at MAX 1.4e-1
    # initialize np array
    arry = []

    Hx = toric_code_x_stabilisers(L)
    Hz = toric_code_z_stabilisers(L)
    logX = toric_code_x_logicals(L)
    logZ = toric_code_z_logicals(L)

    # init error, syndrome, and logicals
    x_noise = (np.random.random(Hx.shape[1]) < percent_error).astype(np.uint8)
    z_noise = (np.random.random(Hz.shape[1]) < percent_error).astype(np.uint8)
    x_syndrome = (Hx @ x_noise) % 2
    z_syndrome = (Hz @ z_noise) % 2
    x_logical = (logX @ x_noise) % 2
    z_logical = (logZ @ z_noise) % 2

    # arrange x syndrome and physical qubits
    x_syndrome = arrange_x_syndrome(x_syndrome)
    x_noise = arrange_physical_qubits(x_noise)
    z_noise = arrange_physical_qubits(z_noise)

    x_syndrome = np.array(x_syndrome)
    z_syndrome = np.array(z_syndrome)
    x_logical = np.array(x_logical)
    z_logical = np.array(z_logical)

    print(x_syndrome.shape)

    observation = np.concatenate((x_syndrome, z_syndrome, x_noise, z_noise, x_logical, z_logical))

    return observation

if __name__ == "__main__":
    # generate random integer:
    randseed = 69

    num_obs =  70000

    seed(randseed)
    print(f"seed: {randseed}")

    L = 17
    percent_error = 0.10 # typical x error rate is at MAX 1.4e-1
    # initialize np array
    arry = []

    Hx = toric_code_x_stabilisers(L)
    Hz = toric_code_z_stabilisers(L)
    logX = toric_code_x_logicals(L)
    logZ = toric_code_z_logicals(L)

    start_time = time.time()

    for i in range(num_obs):
        # init error, syndrome, and logicals
        x_noise = (np.random.random(Hx.shape[1]) < percent_error).astype(np.uint8)
        z_noise = (np.random.random(Hz.shape[1]) < percent_error).astype(np.uint8)
        x_syndrome = (Hx @ x_noise) % 2
        z_syndrome = (Hz @ z_noise) % 2
        x_logical = (logX @ x_noise) % 2
        z_logical = (logZ @ z_noise) % 2

        # arrange x syndrome and physical qubits
        x_syndrome = arrange_x_syndrome(x_syndrome)
        x_noise = arrange_physical_qubits(x_noise)
        z_noise = arrange_physical_qubits(z_noise)

        x_syndrome = np.array(x_syndrome)
        z_syndrome = np.array(z_syndrome)
        x_logical = np.array(x_logical)
        z_logical = np.array(z_logical)

        #print(x_syndrome.shape)

        observation = np.concatenate((x_syndrome, z_syndrome, x_noise, z_noise, x_logical, z_logical))

        #visualize_observation(observation)

        arry.append(observation)

        if i % 10000 == 0:
            print(f'Progress: {i}/{num_obs}')
        
    px_hundo = int(percent_error * 100)
    np.save(f'high-level/test-datasets/HL_data_{L}_{px_hundo}_{num_obs}.npy', arry)

    print(f"Saved dataset with {num_obs} observations, L={L}, and {px_hundo}% error rate in {time.time() - start_time} seconds")