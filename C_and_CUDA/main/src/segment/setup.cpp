#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "../../lib/Segment.h"
#include "../../lib/RealMatrix.h"
extern "C" {
using namespace std;

void computeExteriorPointsAndNormalVectors(RealMatrix x_ext, RealMatrix y_ext, RealMatrix n_x, RealMatrix n_y, Nanostructure nanostructure, int start, int end, double alpha, double leftStep, double rightStep, int leftNum, int rightNum, int n_top, int n_right, int n_bottom, int n_left, double left_x_value, double right_x_value, bool deviceComputation, bool printOutput);
void computeTestPoints(RealMatrix x_test, RealMatrix y_test, Nanostructure nanostructure, int start, int end, double leftStep, double rightStep,int n_top, int n_right, int n_bottom, int n_left, double left_x_value, double right_x_value, bool deviceComputation, bool printOutput);
void computeInteriorPoints(RealMatrix x_int, RealMatrix y_int, RealMatrix x_test, RealMatrix y_test,double alpha, int n_top, int n_right, int n_bottom, int n_left, bool deviceComputation, bool printOutput);

void Segment::setup(Nanostructure nanostructure, int total_grid_points, int num_segments) {
    // Sets up test points and auxiliary points for the segment given a nanostructure
    // that is both on the host and on the device

    // Initialize variables
    //int segment_length = total_grid_points / num_segments;
    int start, end, leftNum, rightNum;
    double leftStep, rightStep, left_f_value, right_f_value, left_x_value, right_x_value, alpha, step;
    //double startvalue, endvalue, step, startstep, endstep, startxvalue, endxvalue, alpha;

    // Determine distance of auxilliary points from nearest test point
    step = nanostructure.x.getHostValue(1) - nanostructure.x.getHostValue(0);
    //alpha = std::min(10e-8,2*std::min(startstep,endstep));

    // Determine wheree the segment starts and ends
    start = current_segment * segment_length;
    //end = min(start + segment_length, total_grid_points);
    end = start + segment_length;
    //printf("(start, end) = (%d, %d)",start,end);

    // Get values at end points
    left_f_value  = nanostructure.f.getHostValue(start);
    right_f_value = nanostructure.f.getHostValue(end - 1);
    left_x_value  = nanostructure.x.getHostValue(start);
    right_x_value = nanostructure.x.getHostValue(end - 1);

    // Determine steps for sides of segment
    //startnum = max(minNumSteps, (int) ceil(startvalue/step));
    //endnum   = max(minNumSteps, (int) ceil(endvalue/step));
    leftNum   = n_left + 1;
    rightNum  = n_right + 1;
    leftStep  = left_f_value/leftNum;
    rightStep = right_f_value/rightNum;
    alpha     = std::min((double)2*step,(double)1.0e-9),2*std::min(leftStep,rightStep);
    alpha *=10;
    
    // Allocate arrays
    /*int n_top       = end - start - 2;
    int n_bottom    = end - start - 2;
    int n_right     = endnum - 1;
    int n_left      = startnum - 1;
    n_int           = n_top + n_right + n_bottom + n_left - 16;
    n_ext           = 2*(end - start - 1) + endnum + startnum;
    n_test          = n_top + n_bottom + n_right + n_left;*/
    
    
    // Compute exterior points
    computeExteriorPointsAndNormalVectors(x_ext, y_ext, n_x, n_y, nanostructure, start, end, alpha, leftStep, rightStep, leftNum, rightNum, n_top, n_right, n_bottom, n_left, left_x_value, right_x_value, deviceComputation, printOutput);
    // Compute test points and temporary test points for interior points
    computeTestPoints(x_test, y_test, nanostructure, start,  end,  leftStep,  rightStep, n_top, n_right, n_bottom, n_left, left_x_value, right_x_value, deviceComputation, printOutput);
    // Compute interior points
    computeInteriorPoints(x_int, y_int, x_test, y_test, alpha, n_top, n_right,  n_bottom,  n_left, deviceComputation, printOutput); 
    
    
    bool save_segment = true;
    if (save_segment) {
        x_test.allocateHost();
        y_test.allocateHost();
        x_ext.allocateHost();
        y_ext.allocateHost();
        x_int.allocateHost();
        y_int.allocateHost();
        n_x.allocateHost();
        n_y.allocateHost();
        x_test.toHost();
        y_test.toHost();
        x_ext.toHost();
        y_ext.toHost();
        x_int.toHost();
        y_int.toHost();
        n_x.toHost();
        n_y.toHost();
        FILE *file;
        char filename[256];
        sprintf(filename,"../../../Data/segments/test_segment_%d.txt", current_segment+1);
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            return;
        }
        for (int k = 0; k < n_test; k++) {
            fprintf(file, "%.4e %.4e\n", x_test.getHostValue(k), y_test.getHostValue(k));
        }
        fclose(file);

        sprintf(filename,"../../../Data/segments/ext_segment_%d.txt", current_segment+1);
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            return;
        }
        for (int k = 0; k < n_ext; k++) {
            fprintf(file, "%.4e %.4e\n", x_ext.getHostValue(k), y_ext.getHostValue(k));
        }
        fclose(file);

        sprintf(filename,"../../../Data/segments/int_segment_%d.txt", current_segment+1);
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            return;
        }
        for (int k = 0; k < n_int; k++) {
            fprintf(file, "%.4e %.4e\n", x_int.getHostValue(k), y_int.getHostValue(k));
        }
        fclose(file);

        sprintf(filename,"../../../Data/segments/n_segment_%d.txt", current_segment+1);
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            return;
        }
        for (int k = 0; k < n_test; k++) {
            fprintf(file, "%.4e %.4e\n", n_x.getHostValue(k), n_y.getHostValue(k));
        }
        fclose(file);

    }


}





}