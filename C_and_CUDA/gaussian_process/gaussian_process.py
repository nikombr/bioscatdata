import ctypes
import numpy as np
from scipy.io import savemat
import os
import time

def gaussian_process(x, y, tau = 1, ell = 1, p=1, device = True, covfunc = "squared_exponential"):
    # covfunc: covariance function
    if covfunc == "squared_exponential":
        type_covfunc = 1
        hyper = np.array([tau,ell])
        var = "tau"
    elif covfunc == "matern":
        type_covfunc = 2
        hyper = np.array([p,ell])
        var = "p"
        tau = p
    else:
        print("Please input a valid covariance function")
        return
    

    so_file = "./gaussian_process.so"
    
    # Compute mesh if we are computing planes
    if len(y) != 0:
        X, Y = np.meshgrid(x, y)
        X = X.flatten()
        Y = Y.flatten()
        dim = 2
    else:
        X = x
        Y = y
        dim = 1
    
    # Get variables for C implementation
    n = len(X)
    X_arr = (ctypes.c_double * len(X))(*X)
    Y_arr = (ctypes.c_double * len(Y))(*Y)
    hyper_arr = (ctypes.c_double * len(hyper))(*hyper)
    dev = 1 if device else 0


    # Get C function
    c_func = ctypes.CDLL(so_file)

    # Execute C implementation
    c_func.gaussian_process(X_arr, Y_arr, n, hyper_arr, 2, dim, dev, type_covfunc)

    # Get result from file
    data = np.loadtxt('../../Data/gaussian_process_realisations/output.txt')
    #print(data)

    # Remove txt output
    os.remove("../../Data/gaussian_process_realisations/output.txt")

    # Save in mat file
    if len(y) == 0:
        mdic = {"data": data, "x": x}
        savemat(f"../../Data/gaussian_process_realisations/curve_{covfunc}_{var}_{tau}_ell_{ell}.mat", mdic)
    else:
        mdic = {"data": data, "x": x, "y": y}
        savemat(f"../../Data/gaussian_process_realisations/plane_{covfunc}_{var}_{tau}_ell_{ell}.mat", mdic)
            
    return data

if __name__ == "__main__":
    x = np.linspace(0,10,300)
    y = np.linspace(0,10,300)
    y = np.array([]);

    covfunc = "matern" # "squared_exponential" "matern"

    if covfunc == "squared_exponential":
        ells = [1, 2, 4];
        taus = [0.25, 0.5, 1];
        for ell in ells:
            for tau in taus:
                Z = gaussian_process(x, y, tau = tau, ell = ell, device = True, covfunc = covfunc)
                time.sleep(2)

    elif covfunc == "matern":

        ells = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 4, 8];
        ps = [1, 2, 4];

        for ell in ells:
            for p in ps:
                Z = gaussian_process(x, y, p = p, ell = ell, device = True, covfunc = covfunc)
                time.sleep(2)
