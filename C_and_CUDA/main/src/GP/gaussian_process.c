#include <stdlib.h>
#include <stdio.h>
#include "../../lib/GP/gaussian_process_inner.h"
#include <cuda_runtime_api.h>

void gaussian_process(double * x, double * y, int n, double * hyper, int num, int dim, int dev, int type_covfunc) {

    gaussian_process_inner(x, y, n, hyper, num, dim, dev, type_covfunc);

}