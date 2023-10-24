# The basis to simulate the toric code: generate grids, simulate errors, check if correction is complete etc

from random import random, seed, randint
import numpy as np
import pandas as pd

# set random seed

#seed(42)

def make_grids(L):
    """TESTED: generate a grid of size LxL
    input:
            L = size of grid
    output:
            stab = LxL grid of stabilizers with 0(no error)
                qubits are between stabilizers
            qubits = 2LxL grid of qubits
            """
    stab1 = [[0 for col in range(L)] for row in range(L)]
    stab2 = [[0 for col in range(L)] for row in range(L)]
    
    qubits1 = [[0 for col in range(L)] for row in range(2 * L)]
    qubits2 = [[0 for col in range(L)] for row in range(2 * L)]
    return stab1, stab2, qubits1, qubits2

def print_grid_stab(grid):
    """TESTED: print the input grid per row, stabilizers """
    print('-' * len(grid[0]))
    for row in grid:
        st = ' '.join([str(n) for n in row])
        print(st)
    print('-' * len(grid[0]))
def print_grid_qubits(grid):
    """TESTED: print the input grid per row, qubits"""
    print('-' * len(grid[0]) * 3)

    for i in range(len(grid)):
        if i % 2 == 0:
            print('+ ' + ' + '.join([str(x) for x in grid[i]]))
        else:
            print('   '.join([str(x) for x in grid[i]]) + '  ')
    print('-' * len(grid[0]) * 3)

def generate_error(x_stab, z_stab, x_error, z_error, px):
    """TESTED: Generate random errors on qubits, and adds 1 to the stabilizers
    if an error on neighbouring qubit occured
    Input:
            grid: LxL grid of stabilizers
            px: probability to have an error on a qubits
    Output:
            grid_s: LxL grid of stabilizers, with values 0-4 for 0-4 errors
                on neighbouring qubits
            grid_q: 2LxL grid of qubits, with values 0-1 for 0 or 1 error on the qubits
            """
    # loop through all qubits:
    for row_idx in range(len(x_error)):
        for col_idx in range(len(x_error[0])):
            error = random() <= px
            #error = 1 if ((row_idx == 9 ) or (col_idx == 4 and row_idx%2==1))else 0
            if not error:
                continue
            if row_idx % 2 == 0: # if even row, then the stabilizer is vert
                stab_row = int(row_idx / 2)
                x_stab[stab_row][col_idx] += 1 
                x_stab[stab_row][col_idx] %= 2
                x_stab[stab_row - 1][col_idx] += 1 
                x_stab[stab_row - 1][col_idx] %= 2
            else: # if odd row, then the stabilizer is horz
                stab_row = int((row_idx - 1) / 2)
                x_stab[stab_row][col_idx] += 1  
                x_stab[stab_row][col_idx] %= 2
                x_stab[stab_row][col_idx - 1] += 1 
                x_stab[stab_row][col_idx - 1] %= 2
            x_error[row_idx][col_idx] += 1

    for row_idx in range(len(z_error)): # there's probably a better way to do this
        for col_idx in range(len(z_error[0])):
            error = 1 if row_idx%2==0 and col_idx%2==0 else 0
            error = 1 if row_idx%4==1 else 0
            #error = x_error[row_idx][col_idx]
            error = random() <= px
            if not error:
                # nothing has to be changed
                continue
            if row_idx % 2 == 0: # if evn, then the stabilizer is horz
                stab_row = int(row_idx / 2)
                z_stab[stab_row][col_idx] += 1 
                z_stab[stab_row][col_idx] %= 2
                z_stab[stab_row][(col_idx + 1) % len(z_stab)] += 1
                z_stab[stab_row][(col_idx + 1) % len(z_stab)] %= 2
            else: # if odd, then the stabilizer is vert
                stab_row = int((row_idx - 1) / 2)
                z_stab[stab_row][col_idx] += 1 
                z_stab[stab_row][col_idx] %= 2
                z_stab[(stab_row + 1) % len(z_stab)][col_idx] += 1 
                z_stab[(stab_row + 1) % len(z_stab)][col_idx] %= 2
            z_error[row_idx][col_idx] += 1
    return x_stab, z_stab, x_error, z_error


from Datavis import pp_code, visualize_observation

if __name__ == "__main__":
    # generate random integer:
    randseed = 69

    num_obs =  70000

    seed(randseed)
    print(randseed)
    L = 5
    px = 0.100 # typical x error rate is at MAX 1.4e-1
    # initialize np array
    arry = []
    for i in range(num_obs):
        x_stab, z_stab, x_qubits, z_qubits = make_grids(L)
        x_stab, z_stab, x_qubits, z_qubits = generate_error(x_stab, z_stab, x_qubits, z_qubits, px)
        """ 

        print_grid_stab(x_stab)
        print_grid_qubits(x_qubits)

        print_grid_stab(z_stab)
        print_grid_qubits(z_qubits) """

        x_stab = np.array(x_stab)
        x_qubits = np.array(x_qubits)
        z_stab = np.array(z_stab)
        z_qubits = np.array(z_qubits)

        flat_x_stab = x_stab.flatten()
        flat_z_stab = z_stab.flatten()
        flat_x_qubits = x_qubits.flatten()
        flat_z_qubits = z_qubits.flatten()

        observation = np.concatenate((flat_x_stab, flat_z_stab, flat_x_qubits, flat_z_qubits))

        #visualize_observation(observation)

        arry.append(observation)

        if i % 10000 == 0:
            print(f'Progress: {i}/{num_obs}')
        
        

        #pp_code(x_stab, z_stab, x_qubits, z_qubits)
    # save data as a numpy array    
    pxhundo = int(px * 100)
    np.save(f'low-level/test_datasets/MC_data_{L}_{pxhundo}_{num_obs}.npy', arry)