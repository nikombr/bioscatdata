import ctypes
import numpy as np
from scipy.io import savemat
import os
import time
import matplotlib.pyplot as plt
import math
import shutil

def executeInverse(num_segments = 1, delta = 0.01, maxiter = 10, total_grid_points=100,protein_structure = "demoleus2x2",tau = 1, ell = 1, p=1, covfunc = "squared_exponential"): # "Retinin2x2" or "demoleus2x2"    
    # covfunc: covariance function
    if covfunc == "squared_exponential":
        type_covfunc = 1
        hyper = np.array([tau,ell])
        var = "tau"
        filename = f"{protein_structure}/squared_exponential/tau_{tau}_ell_{ell}_delta_{delta}"
    elif covfunc == "matern":
        type_covfunc = 2
        hyper = np.array([p,ell])
        var = "p"
        tau = p
        filename = f"{protein_structure}/matern/p_{p}_ell_{ell}_delta_{delta}"
    else:
        print("Please input a valid covariance function")
        return

    hyper_arr = (ctypes.c_double * len(hyper))(*hyper)

    so_file = "./so/inverse.so"

    # Get C function
    c_func = ctypes.CDLL(so_file)
    protein_structure_encoded = protein_structure.encode('utf-8')

    # Execute C implementation
    c_func.inverse(protein_structure_encoded, num_segments, total_grid_points, hyper_arr, 2, type_covfunc, ctypes.c_double(delta), maxiter, filename.encode('utf-8'))

    plt.figure()
    for i in range(num_segments):
        filename = f'../../../Data/segments/test_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1],'r.-')
  

    
    for i in range(num_segments):
        filename = f'../../../Data/segments/ext_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1],'b.-')
    


    for i in range(num_segments):
        filename = f'../../../Data/segments/int_segment_{i+1}.txt'
        data = np.loadtxt(filename)
        plt.plot(data[:,0],data[:,1],'m.-')
    plt.savefig('plots/points_inverse.png')
    plt.close()
    filename = "../../../Results/inverse/output.txt"
    data = np.loadtxt(filename);

    plt.figure()
    [m, n] = data.shape
    for i in range(m):
        plt.plot(data[i,:]+3e-8)
    plt.savefig("plots/inverse.png")
    plt.close()

    plt.figure()
    [m, n] = data.shape
    for i in range(m):
        plt.plot(data[-1,:]+3e-8)
    plt.savefig("plots/inverse_last.png")
    plt.close()

if __name__ == "__main__":

    executeInverse(p=4,ell=0.5,total_grid_points=300,num_segments=1,delta=0.0001,maxiter=10,covfunc = "squared_exponential") # squared_exponential matern
    
    
    

    
    


