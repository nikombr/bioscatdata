#ifndef _CONSTANTS_H
#define _CONSTANTS_H
extern "C" {

struct Constants {

    // Mathematical constants
    double pi = 3.14159265358979323846;

    // Physical constants
    double eta0     = 377.0;            // 377 Ohm, impedance of free space
    double n0       = 1.0;              // Refractive index of air
    double epsilon0 = 8.8541878188e-12; // Permittivity of free space
    double mu0      = 1.25663706127e-6; // Permeability of free space

    // Chosen constants (I have used the same constants as Mirza for some of them)
    double lambda0 = 325e-9;    // Default value of wavelength in free space
    double ns    = 1.6;         // Refractive index of the substrate %0.571 - 1i*0.636;
    double n1    = 1.6;         // Refractive index of the nanowire/nanostructure % 5.05506 - 1i*3.20418;
    double alpha = 0.86;        // Scaling for placement of auxiliary sources for the nanowires

    // Computed constants
    double Gamma_ref;       // Fresnel reflection coefficient
    double Gamma_trans;     // Fresnel transmission coefficient
    double k0;              // Wave number in free space
    double ks;              // Wave number in substrate
    double k1;              // Wave number in nanowire/nanostructure

    Constants() {
        // Computed constants
        Gamma_ref   = (n0 - ns)/(n0 + ns);  // Fresnel reflection coefficient
        Gamma_trans = 2*n0/(n0 + ns);       // Fresnel transmission coefficient
        k0          = 2*pi/lambda0;         // Wave number in free space
        ks          = k0*ns/n0;             // Wave number in substrate
        k1          = k0*n1/n0;             // Wave number in nanowire/nanostructure
        //printf("Constants initialized!\n");
    }

    void newWavelength(double lambda) {
        lambda0 = lambda;
        // Computed constants
        Gamma_ref   = (n0 - ns)/(n0 + ns);  // Fresnel reflection coefficient
        Gamma_trans = 2*n0/(n0 + ns);       // Fresnel transmission coefficient
        k0          = 2*pi/lambda0;         // Wave number in free space
        ks          = k0*ns/n0;             // Wave number in substrate
        k1          = k0*n1/n0;             // Wave number in nanowire/nanostructure
        //printf("Constants initialized with new wavelength!\n");
    }


};

}
#endif
