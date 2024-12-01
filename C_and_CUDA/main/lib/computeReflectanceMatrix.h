
#ifndef _COMPUTE_REFLECTANCE_MATRIX_H
#define _COMPUTE_REFLECTANCE_MATRIX_H

#include "RealMatrix.h"
extern "C" {
void computeReflectanceMatrix(Nanostructure proposedNanostructure, BioScat bioscat, RealMatrix reflectance, RealMatrix lambdas, RealMatrix betas, int num_lambdas, int num_betas, int n_obs, bool deviceComputation);
void computeReflectanceMatrixGetData(BioScat bioscat, RealMatrix reflectance, double * lambdas, double * betas, int num_lambdas, int num_betas, int n_obs, bool deviceComputation);
}
#endif