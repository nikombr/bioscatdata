#include "../lib/inverse.h"

void executeInverse(char* protein_structure, int num_segments, int total_grid_points, double * hyper, int num, int type_covfunc, double delta, int maxiter, char * filename) {

    inverse(protein_structure, num_segments, total_grid_points, hyper, num, type_covfunc, delta, maxiter, filename);

}