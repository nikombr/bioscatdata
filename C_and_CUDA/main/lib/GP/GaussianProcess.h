#ifndef _INVERSE_PLANE_H
#define _INVERSE_PLANE_H
// x_h, y_h: x and y coordinates for measurement points on host
    // n: number of measurement points
    // hyper: array of hyperparameters
    // num: number of hyperparameters
class GaussianProcess {
    private:        
        int n;              // number of points for estimating plane
        double *hyper_h;    // hyperparameters on host
        double *hyper_d;    // hyperparameters on device
        int num;            // number of hyperparameters
        int dim;            // dimension of the problem, either 1 for curve or 2 for plane
        double *x_h;        // x coordinates for estimating plane on host
        double *x_d;        // x coordinates for estimating plane on device
        double *y_h;        // y coordinates for estimating plane on host
        double *y_d;        // y coordinates for estimating plane on device
        double **M_d;       // covariance matrix and later lower triangular matrix from Cholesky factorization on device
        double **M_inv_d;   // covariance matrix and later inverse of covariance matrix on the device
        int type_covfunc;
    public:
        double **M_h;       // covariance matrix and later lower triangular matrix from Cholesky factorization on host
        double **M_inv_h;   // covariance matrix and later inverse of covariance matrix on the host
        double *M_log;      // M_d[0] on device
        double *M_inv_log;  // M_inv_d[0] on device
        bool device;
        double *p_h;        // random vector and later height of plane in location (x,y) on host
        double *p_d;        // random vector and later height of plane in location (x,y) on device
        double logDeterminant;
        GaussianProcess();
        GaussianProcess(double* x_h, double* y_h, int n, double* hyper, int num, int dim, int dev, int type_covfunc);  // Constructer, sets default values and allocates
        void free();                                                                // Destructer
        void covariance_matrix();                                                           // Computes covariance matrix K
        void cholesky();                                                                    // Does cholesky factorization of K to compute L
        void generate_random_vector();                                                      // Generates random vector p
        void realisation();                                                                 // Computes realisation of the Gaussian process from L
        GaussianProcess(int n, double* hyper, int num, int type_covfunc);
        void compute_inverse();
        double compute_prior(double * f_d);
};

#endif