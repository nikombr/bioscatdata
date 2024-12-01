#ifndef _REAL_MATRIX_H
#define _REAL_MATRIX_H
extern "C" {
class RealMatrix {
    // Stores a real matrix rowmajor in a vector both on the host and the device
    private: // Private to avoid wrong indexing
        double * val_h; // Entries on host
        double * val_d; // Entries on device
    public:
        int rows; // Number of rows
        int cols; // Number of cols
        int depth;
        bool host;
        bool device;

        // Methods
        RealMatrix() {
            rows = 0;
            cols = 0;
            depth = 0;
            val_h = NULL;
            val_d = NULL;
        };  // Constructer
        RealMatrix(int rows);                                       // Constructer, allocates vectors and initializes to zero
        RealMatrix(int rows, int cols);                             // Constructer, matrices arrays and initializes to zero
        RealMatrix(int rows, int cols, int depth);
        RealMatrix(int rows, bool host, bool device);                                       // Constructer, allocates vectors and initializes to zero
        RealMatrix(int rows, int cols, bool host, bool device);                             // Constructer, matrices arrays and initializes to zero
        RealMatrix(int rows, int cols, int depth, bool host, bool device);
        void allocateHost() {
            this->host = true;
            // Allocate vectors on host
            cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
            if (val_h == NULL) {
                fprintf(stderr, "Memory allocation failed.\n");
                return;
            }
        }
        void free();                                                // Frees arrays
        void toHost();                                              // Sends data from device to host
        void toDevice();                                            // Sends data from host to device
        void setHostValue(int r, double val);                       // Sets host value for vectors
        void setHostValue(int r, int c, double val);                // Sets host value for matrices
        void setHostValue(int r, int c, int d, double val);
        __device__ void setDeviceValue(int r, double val) {         // Sets device value for vectors
            val_d[r] = val;
        }          
        __device__ void setDeviceValue(int r, int c, double val) {  // Sets device value for matrices
            val_d[r*cols + c] = val;
        }   
        __device__ void setDeviceValue(int r, int c, int d, double val) {
            val_d[r * (cols * depth) + c * depth + d] = val;
        }

        double getHostValue(int r);                                 // Gets host value for vectors
        double getHostValue(int r, int c);                          // Gets host value for matrices
        double getHostValue(int r, int c, int d);
        __device__ double getDeviceValue(int r) {                   // Gets device value for vectors
            return val_d[r];
        }                    
        __device__ double getDeviceValue(int r, int c) {            // Gets device value for matrices
            return val_d[r*cols + c];
        }            
        __device__ double getDeviceValue(int r, int c, int d) {
            return val_d[r * (cols * depth) + c * depth + d];
        }
        double * getHostPointer() {
            return val_h;
        }
        double * getDevicePointer() {
            return val_d;
        }
        void setHostPointer(double * val) {
            val_h = val;
        }
        void setDevicePointer(double * val) {
            val_d = val;
        }
        void dumpVector(char * filename) {
            FILE *file;
            file = fopen(filename, "w");
            if (file == NULL) {
                perror("Error opening file");
                printf("File: %s\n",filename);
                return;
            }
            for (int i = 0; i < rows*cols*depth; i++) {
                fprintf(file, "%e\n", val_h[i]);
            }
            fclose(file);
        }

        void loadVector(char * filename) {
            FILE *file;
            file = fopen(filename, "r");
            if (file == NULL) {
                perror("Error opening file");
                printf("File: %s\n",filename);
                return;
            }
            for (int i = 0; i < rows*cols*depth; i++) {
                fscanf(file, "%lf\n", &val_h[i]);  // Reading each value into the array
            }
            fclose(file);
        }

        void print() {
            if (depth == 1) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        printf("%e ",getHostValue(i, j));
                    }
                    printf("\n");
                }
            }
        }

        double findMin() {
            double minimum = 10;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    minimum = std::min(getHostValue(i, j), minimum);
                }        
            }
            return minimum;
        }
        void dumpResult(const char * filename);

  
};
}

#endif