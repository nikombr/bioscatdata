#ifndef __CUDA_MALLOC_2D
#define __CUDA_MALLOC_2D

void host_malloc_2d(double ***B, int n);
void device_malloc_2d(double ***B, double **b, int n);

#define HAS_FREE_2D
void host_free_2d(double **B);
void device_free_2d(double **B, double *b);

#endif
