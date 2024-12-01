#ifndef _COMPLEX_MATRIX_H
#define _COMPLEX_MATRIX_H

class ComplexMatrix {
    // Stores a imagMatrix rowmajor in a vector both on the host and the device
    private: // Private to avoid wrong indexing
        double * real_h;    // Real entries on host
        double * imag_h;    // Imag entries on host
        double * real_d;    // Real entries on device
        double * imag_d;    // Imag entries on device
    public:
        int    rows; // Number of rows
        int    cols; // Number of cols
        bool host;
        bool device;

        // Methods
        ComplexMatrix() {
            real_h = NULL;
            imag_h = NULL;
            real_d = NULL;
            imag_d = NULL;
        } 
        ComplexMatrix(int rows, int cols);                               // Constructer, allocates arrays and initializes to zero
        ComplexMatrix(int rows);
        ComplexMatrix(int rows, int cols, bool host, bool device);                               // Constructer, allocates arrays and initializes to zero
        ComplexMatrix(int rows, bool host, bool device);
        void free();                                                    // Frees arrays
        void toHost();                                                  // Sends data to host
        void toDevice();                                                // Sends data to device
        void setHostRealValue(int r, double val);                       // Sets host real value for vectors
        void setHostRealValue(int r, int c, double val);                // Sets host real value for matrices
        void setHostImagValue(int r, double val);                       // Sets host imag value for vectors
        void setHostImagValue(int r, int c, double val);                // Sets host imag value for matrices
        __device__ void setDeviceRealValue(int r, double val) {         // Sets device real value for vectors
            real_d[r] = val;
        }              
        __device__ void setDeviceRealValue(int r, int c, double val) {  // Sets device real value for matrices
            real_d[r*cols + c] = val;
        }       
        __device__ void setDeviceImagValue(int r, double val) {         // Sets device imag value for vectors
            imag_d[r] = val;
        }          
        __device__ void setDeviceImagValue(int r, int c, double val) {  // Sets device imag value for matrices
            imag_d[r*cols + c] = val;
        }   
        double getHostRealValue(int r);                                 // Gets host real value for vectors
        double getHostRealValue(int r, int c);                          // Gets host real value for matrices
        double getHostImagValue(int r);                                 // Gets host imag value for vectors
        double getHostImagValue(int r, int c);                          // Gets host imag value for matrices
        __device__ double getDeviceRealValue(int r) {                   // Gets device real value for vectors
            return real_d[r];
        }                        
        __device__ double getDeviceRealValue(int r, int c) {            // Gets device real value for matrices
            return real_d[r*cols + c];
        }                 
        __device__ double getDeviceImagValue(int r) {                   // Gets device imag value for vectors
            return imag_d[r];
        }                     
        __device__ double getDeviceImagValue(int r, int c) {            // Gets device imag value for matrices
            return imag_d[r*cols + c];
        } 
        void dumpResult(const char * filename);    

        void setHostZero() {
            memset(real_h, 0, rows*cols*sizeof(double));
            memset(imag_h, 0, rows*cols*sizeof(double));
            
        }   
        void setDeviceZero() {
            cudaMemset(real_d, 0, rows*cols*sizeof(double));
            cudaMemset(imag_d, 0, rows*cols*sizeof(double));
            cudaDeviceSynchronize();
        }    
        double * getDeviceRealPointer() {
            return real_d;
        }
        double * getDeviceImagPointer() {
            return imag_d;
        }





};


#endif