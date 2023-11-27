import numpy as np
import matplotlib.pyplot as plt


def pp_code(x_stab, z_stab, x_qubits, z_qubits):
    L = x_qubits.shape[0]/2
    # initialize empty plot
    fig, ax = plt.subplots(figsize=(2*L,2*L))
    # set bounds for image
    ax.set_xlim(-1, L+1)
    ax.set_ylim(-1, L+1)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    # remove gridlines to the left of x = 0
    for down, row in enumerate(x_qubits): # AWFUL AWFUL COORDINATE SYSTEM
        for right, bit in enumerate(row):
            #print(f"down: {down}, right: {right}, bit: {bit}")
            if down%2 == 0:
                ax.plot(right+0.5, L-(down*0.5), marker='o', color='k', markersize=5)
                if bit == 1:
                    ax.plot(right+0.5, L-(down*0.5), marker='o', color='r', markersize=5)
                elif z_qubits[down][right] == 1:
                    ax.plot(right+0.5, L-(down*0.5), marker='o', color='y', markersize=5)
                if z_qubits[down][right] == 1 and bit == 1:
                    ax.plot(right+0.5, L-(down*0.5), marker='o', color='g', markersize=5)
            else:
                ax.plot(right, L-(down*0.5), marker='o', color='k', markersize=5)
                if bit == 1:
                    ax.plot(right, L-(down*0.5), marker='o', color='r', markersize=5)
                elif z_qubits[down][right] == 1:
                    ax.plot(right, L-(down*0.5), marker='o', color='y', markersize=5)
                if z_qubits[down][right] == 1 and bit == 1:
                    ax.plot(right, L-(down*0.5), marker='o', color='g', markersize=5)
    #print("drawing errors")
    for down, row in enumerate(x_stab):
        for right, bit in enumerate(row):
            #print(f"down: {down}, right: {right}, bit: {bit}")
            if bit == 1:
                ax.plot(right+0.5, L-(down+0.5), marker='o', color='b', markersize=10)
    for down, row in enumerate(z_stab):
        for right, bit in enumerate(row):
            #print(f"down: {down}, right: {right}, bit: {bit}")
            if bit == 1:
                ax.plot((right + 0)% L, L-((down+0) %L ), marker='o', color='b', markersize=10)

    plt.show()
    
def visualize_observation(observation):
    # number of entries in observation
    six_n_squared = len(observation)
    n_squared = int(six_n_squared / 6)
    n = int(n_squared ** 0.5)

    print(f"distance: {n}")

    # first n^2 entries are x_stab
    x_stab = observation[:n_squared]
    # next n^2 entries are z_stab
    z_stab = observation[n_squared:2*n_squared]
    # next 2*n^2 entries are x_qubits
    x_qubits = observation[2*n_squared:4*n_squared]
    # next 2*n^2 entries are z_qubits
    z_qubits = observation[4*n_squared:]

    # reshape into 2D arrays
    x_stab = np.reshape(x_stab, (n, n))
    z_stab = np.reshape(z_stab, (n, n))
    x_qubits = np.reshape(x_qubits, (2*n, n))
    z_qubits = np.reshape(z_qubits, (2*n, n))

    # plot 
    pp_code(x_stab, z_stab, x_qubits, z_qubits)
