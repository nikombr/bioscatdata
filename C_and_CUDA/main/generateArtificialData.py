import ctypes
import numpy as np
from scipy.io import savemat
import os
import time
import matplotlib.pyplot as plt
import math
import shutil
import time

def executeGenerateArtificialData(num_obs_points=100, num_segments = 1, total_grid_points=100,protein_structure = "demoleus2x2"): # "Retinin2x2" or "demoleus2x2"

    if not (total_grid_points % num_segments == 0):
        print(f"Total number of grid points was changed from {total_grid_points} to {int(np.ceil(total_grid_points/num_segments)*num_segments)} to make sure that each segment has the same amount of points.")
        total_grid_points = int(np.ceil(total_grid_points/num_segments)*num_segments)

    phi = np.linspace(0,math.pi,num_obs_points);
    x = 10**(-2)*np.cos(phi)
    y = 10**(-2)*np.sin(phi)

    directory_name = f'../../../Data/artificial_data/{protein_structure}/num_segments_{num_segments}_total_grid_points_{total_grid_points}/'
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
        os.makedirs(directory_name)  # Recreate the directory if you want to keep it
    else:
        os.mkdir(directory_name)

    # Prepare observation points
    n = len(x)
    x_arr = (ctypes.c_double * n)(*x)
    y_arr = (ctypes.c_double * n)(*y)
    

    so_file = "./so/generateArtificialData.so"

    # Get C function
    c_func = ctypes.CDLL(so_file)
    protein_structure_encoded = protein_structure.encode('utf-8')

    lambda0 = 325e-9;
    lambdas = np.linspace(0.5*lambda0,1.5*lambda0,10);
    betas = np.linspace(math.pi/4,3*math.pi/2,10); # der er noget galt med 3D matrix index

    lambdas_arr = (ctypes.c_double * len(lambdas))(*lambdas)
    betas_arr = (ctypes.c_double * len(betas))(*betas)

    # Execute C implementation
    c_func.executeGenerateArtificialData(x_arr, y_arr, n, protein_structure_encoded, num_segments, total_grid_points,betas_arr, lambdas_arr,len(betas),len(lambdas))
    
    # Find files and move them to the correct directory
    source = '../../../Data/artificial_data/temp/reflectance.txt'
    shutil.move(source, directory_name)
    
    np.savetxt(f'{directory_name}x_obs.txt', x, fmt='%e', delimiter='\n')
    np.savetxt(f'{directory_name}y_obs.txt', y, fmt='%e', delimiter='\n')
    np.savetxt(f'{directory_name}lambdas.txt', lambdas, fmt='%e', delimiter='\n')
    np.savetxt(f'{directory_name}betas.txt', betas, fmt='%e', delimiter='\n')
    """
    plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/test_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1])
    plt.savefig('plots/test_points.png')
    plt.close()

    plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/ext_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1])
    plt.savefig('plots/ext_points.png')
    plt.close()

    plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/int_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1])
    plt.savefig('plots/int_points.png')
    plt.close()

    data = 0;
    return data
    """

if __name__ == "__main__":
    
    
    for num_segments in [1, 2, 4, 5]:
        executeGenerateArtificialData(total_grid_points=300,num_segments=num_segments)
        time.sleep(1)
    
    
    

    
    


