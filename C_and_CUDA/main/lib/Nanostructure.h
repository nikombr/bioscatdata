#ifndef _NANOSTRUCTURE_H
#define _NANOSTRUCTURE_H
extern "C" {
#include "RealMatrix.h"

#ifdef _MAKE_2D

struct Nanostructure {
    RealMatrix f; // Function value in x
    RealMatrix x; // x-values

    // Constructor 
    Nanostructure() {
        f = RealMatrix();
        x = RealMatrix();
        //std::cout << "Empty segment constructor." << std::endl;
    } 
    Nanostructure(int n) {
        f = RealMatrix(n, 1);
        x = RealMatrix(n, 1);
        //std::cout << "Non-empty segment constructor." << std::endl;
    }
};

#endif

#ifdef _MAKE_3D

struct Nanostructure {
    RealMatrix f; // Function value in (x,y)
    RealMatrix x; // x-values
    RealMatrix y; // y-values
};

#endif




}

#endif