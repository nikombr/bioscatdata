import ctypes
import numpy as np
from scipy.io import savemat
import os
import time
import matplotlib.pyplot as plt
import math

def executeFarFieldPattern(phi, total_grid_points=100,num_segments = 1, protein_structure = "demoleus2x2", beta = 0, lambd = 325e-9, deviceComputation = False): # "Retinin2x2" or "demoleus2x2"

    # Prepare observation points
    n = len(phi)
    phi_arr = (ctypes.c_double * n)(*phi)
    
    so_file = "./so/farFieldPattern.so"

    # Get C function
    c_func = ctypes.CDLL(so_file)
    protein_structure_encoded = protein_structure.encode('utf-8')

    # Execute C implementation
    c_func.executeFarFieldPattern(phi_arr, n, protein_structure_encoded, num_segments,total_grid_points, ctypes.c_double(beta*math.pi/180), ctypes.c_double(lambd), int(deviceComputation))

    mdic = dict();
    mdic['phi'] = phi

    filename = f'../../../Results/forward/farFieldPattern.txt'
    F = np.loadtxt(filename)
    print(F.shape)
    plt.figure()
    plt.polar(phi, F)
    plt.savefig(f'plots/farFieldPattern.png')
    plt.close()
    #os.remove(filename)

    location = 'far_field_pattern';
    savename = f'../../../Results/forward/{protein_structure}/{location}/absolute_field_beta_{int(beta)}_lambda_{int(lambd*10**9)}_num_segments_{int(num_segments)}_total_grid_points_{int(total_grid_points)}.mat'
    savemat(savename, mdic)


    plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/test_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1],'k.')
    #plt.savefig('plots/test_points.png')

    #plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/ext_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1],'.')
    #plt.savefig('plots/ext_points.png')

    #plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/int_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1],'.')
    plt.savefig('plots/all_points.png')

def getData():
    obs_grid = 200;
    Y = np.linspace(0,21*10**(-7),obs_grid);
    location = "far"; # near, far, (far_field_pattern)
    #Y = Y + 4.38059442329516e-08;
    Y = Y + 3e-2;
    X = np.linspace(-10.5*10**(-7),10.5*10**(-7),obs_grid);

    grid_sizes = [100, 300, 500, 1000];
    
    for protein_structure in ["demoleus2x2", "Retinin2x2"]: # "Retinin2x2" or "demoleus2x2"
        for beta in [0, 90]:
            for n in grid_sizes:
                executeForward(x = X, y = Y, num_segments = 1, beta = beta, total_grid_points=n, protein_structure = protein_structure, deviceComputation = True, location = location)


def getFarFieldPattern():
    obs_grid = 200;
    phi = np.linspace(0,np.pi,obs_grid);
    r = 3e-2; # 3 cm
    X = np.cos(phi)*r
    Y = np.sin(phi)*r
    location = "far_field_pattern"
    grid_sizes = [100, 300, 500, 1000];
    
    for protein_structure in ["demoleus2x2", "Retinin2x2"]: # "Retinin2x2" or "demoleus2x2"
        for beta in [0, 90]:
            for n in grid_sizes:
                executeForward(x = X, y = Y, num_segments = 1, beta = beta, total_grid_points=n, protein_structure = protein_structure, deviceComputation = True, location = location)

if __name__ == "__main__":
    
    obs_grid = 200;
    phi = np.linspace(0,np.pi,obs_grid);
    location = "far_field_pattern"
    
    protein_structure = ["demoleus2x2", "Retinin2x2"] # "Retinin2x2" or "demoleus2x2"

    executeFarFieldPattern(phi=phi, num_segments = 1, beta = 0, total_grid_points=300, protein_structure = protein_structure[0], deviceComputation = True)



