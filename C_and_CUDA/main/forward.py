import ctypes
import numpy as np
from scipy.io import savemat
import os
import time
import matplotlib.pyplot as plt
import math

def executeForward(x, y, total_grid_points=100,num_segments = 1, protein_structure = "demoleus2x2", beta = 0, lambd = 325e-9, deviceComputation = False, location = "near"): # "Retinin2x2" or "demoleus2x2"


    nx = len(x);
    ny = len(y);
    if location == "far_field_pattern":
        Xmesh = x;
        Ymesh = y;
    else:
        Xmesh, Ymesh = np.meshgrid(x,y)
        nx, ny = Xmesh.shape
        #print(nx,ny)
        #print(Xmesh)
        #print(Ymesh)
        Xmesh = Xmesh.flatten()
        Ymesh = Ymesh.flatten()

    # Prepare observation points
    n = len(Xmesh)
    x_arr = (ctypes.c_double * n)(*Xmesh)
    y_arr = (ctypes.c_double * n)(*Ymesh)
    

    so_file = "./so/forward.so"

    # Get C function
    c_func = ctypes.CDLL(so_file)
    protein_structure_encoded = protein_structure.encode('utf-8')

    # Execute C implementation
    c_func.executeForward(x_arr, y_arr, n, protein_structure_encoded, num_segments,total_grid_points, ctypes.c_double(beta*math.pi/180), ctypes.c_double(lambd), int(deviceComputation))

    variables = ['x', 'y', 'z'];
    types = ['scat', 'inc']
    field_types = ['E','H']

    mdic = dict();
    mdic['x'] = x
    mdic['y'] = y

    for j in range(2):
        typ = types[j]
        for k in range(2):
            field_typ = field_types[k]
            if location == "far_field_pattern":
                fields = np.zeros((nx,3),dtype=complex)
            else:
                fields = np.zeros((nx,ny,3),dtype=complex)

            for i in range(3):
                var = variables[i]
                filename = f'../../../Results/forward/{field_typ}{var}_{typ}.txt'
                data = np.loadtxt(filename)
                field = data[:,0] + 1j*data[:,1]
                print(field.shape)
                if location == "far_field_pattern":
                    fields[:,i] = field
                else:
                    fields[:,:,i] = np.reshape(field,[nx,ny])

                    plt.figure()
                    plt.imshow(np.abs(fields[:,:,i]))
                    plt.colorbar()
                    plt.savefig(f'plots/{field_typ}{var}_{typ}.png')
                    plt.close()
                    os.remove(filename)
                    
            print(fields.shape)
            mdic[f'{field_typ}_{typ}'] = fields

    savename = f'../../../Results/forward/{protein_structure}/{location}/fields_beta_{int(beta)}_lambda_{int(lambd*10**9)}_num_segments_{int(num_segments)}_total_grid_points_{int(total_grid_points)}.mat'
    savemat(savename, mdic)

    n = 50;
    plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/test_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:(n-2),0],data[:(n-2),1],'k.')
    for i in range(num_segments):
        filename = f'../../../Data/segments/n_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        #plt.plot(data[:10,0],data[:10,1],'k.')
    #plt.savefig('plots/test_points.png')

    #plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/ext_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:(n),0],data[:(n),1],'.')
    #plt.savefig('plots/ext_points.png')

    #plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/int_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:(n-6),0],data[:(n-6),1],'.')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig('plots/all_points.png')

    filename = f'../../../Data/segments/test_segment_1.txt'
    data = np.loadtxt(filename)

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
    Y = np.linspace(-10*10**(-7),31*10**(-7),obs_grid);
    location = "near"; # near, far, (far_field_pattern)
    #Y = Y + 1.6e-6;
    #Y = Y + 3e-2;
    X = np.linspace(-20.5*10**(-7),20.5*10**(-7),obs_grid);
    protein_structure = "demoleus2x2"  # "Retinin2x2" or "demoleus2x2"
    
    executeForward(x = X, y = Y, num_segments = 1, beta = 0, total_grid_points=300, protein_structure = protein_structure, deviceComputation = True, location = location)
    


