#ifndef _SEGMENT_H
#define _SEGMENT_H
#include "cuSolver.h"
extern "C" {
#include "RealMatrix.h"
#include "ComplexMatrix.h"
#include "Nanostructure.h"
#include "Field.h"
#include "Constants.h"


class Segment {
    public:
        RealMatrix x_int;
        RealMatrix y_int;
        RealMatrix x_ext;
        RealMatrix y_ext;
        RealMatrix x_test;
        RealMatrix y_test;
        RealMatrix n_x;
        RealMatrix n_y;
        ComplexMatrix C; // Partial solution of the linear system
        ComplexMatrix D; // Partial solution of the linear system
        RealMatrix A; // Linear system matrix in Ax=b
        RealMatrix b; // Linear system vector in Ax=b
        Field E_scat_matrix;
        Field H_scat_matrix;
        Field E_scat;
        ComplexMatrix F; // Far field pattern
        Field H_scat;
        Field E_int_matrix; 
        Field H_int_matrix; 
        Field E_inc_vector;
        Field H_inc_vector;
        Field E_ref_vector;
        Field H_ref_vector;
        int polarisation = 1; // Either 1 or 2
        Constants constants;
        int n_ext, n_int, n_test, n_obs, n_top, n_right, n_bottom, n_left; // Number of points
        int segment_length;
        int minNumSteps; // Minimum number of steps for sides of segment
        bool deviceComputation = false;
        int current_segment;
        bool printOutput = false;
        cusolverDnHandle_t handle;
        double *A_T_d, *x_d;



        Segment(); // Empty constructor
        void allocate(); // Allocation of matrices
        void free(); // Free points
        void freeScatteredFields(); // Free fields
        void freeScatteredSubFields(); // Free fields
        void freeInteriorFields(); // Free fields
        void freeIncidentFields(); // Free fields
        void freeReflectedFields(); // Free fields
        void setup(Nanostructure nanostructure, int total_grid_points, int num_segments); // Setup segment
        void computeIncidentFieldVectors(RealMatrix y); // Computes vectors in observation points
        void computeReflectedFieldVectors(RealMatrix y); // Computes vectors in observation points
        void computeScatteredFieldMatrices(RealMatrix x, RealMatrix y, bool far_field_approximation);
        void computeInteriorFieldMatrices(RealMatrix x, RealMatrix y);
        void computeTotalFields();
        void setupRightHandSide();
        void computeFieldsForLinearSystem(); // Computes vectors and matrices in test points
        void setupSystemSubMatrices();
        void solveLinearSystem();
        void setupSystemMatrix();
        void newWavelength(double lambda);
        void computeScatteredSubFields();
        void computeFarFieldPattern(RealMatrix phi);
};

}

#endif