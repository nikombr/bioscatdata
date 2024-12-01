#ifndef _FIELD_H
#define _FIELD_H
extern "C" {
#include "ComplexMatrix.h"
struct Field {
    // Stores a real matrix rowmajor in a vector both on the host and the device
    public:
        ComplexMatrix x;
        ComplexMatrix y;
        ComplexMatrix z;
        int rows;
        int cols;
        bool x_bool = false;
        bool y_bool = false;
        bool z_bool = false;

        // Methods
        Field() {
            x = ComplexMatrix();
            y = ComplexMatrix();
            z = ComplexMatrix();
        };  // Constructer
        Field(int rows) {
            x_bool = true;
            y_bool = true;
            z_bool = true;
            this->rows = rows;
            this->cols = 1;
            x = ComplexMatrix(rows);
            y = ComplexMatrix(rows);
            z = ComplexMatrix(rows);
        };  // Constructer

        Field(int rows, bool host, bool device) {
            x_bool = true;
            y_bool = true;
            z_bool = true;
            this->rows = rows;
            this->cols = 1;
            x = ComplexMatrix(rows, host, device);
            y = ComplexMatrix(rows, host, device);
            z = ComplexMatrix(rows, host, device);
        };  // Constructer

        Field(int rows, bool x_bool_new, bool y_bool_new, bool z_bool_new) {
            x_bool = x_bool_new;
            y_bool = y_bool_new;
            z_bool = z_bool_new;
            this->rows = rows;
            this->cols = 1;
            if (x_bool) x = ComplexMatrix(rows);
            if (y_bool) y = ComplexMatrix(rows);
            if (z_bool) z = ComplexMatrix(rows);
        };  // Constructer
        Field(int rows, bool x_bool_new, bool y_bool_new, bool z_bool_new, bool host, bool device) {
            x_bool = x_bool_new;
            y_bool = y_bool_new;
            z_bool = z_bool_new;
            this->rows = rows;
            this->cols = 1;
            if (x_bool) x = ComplexMatrix(rows, host, device);
            if (y_bool) y = ComplexMatrix(rows, host, device);
            if (z_bool) z = ComplexMatrix(rows, host, device);
        };  // Constructer
        Field(int rows, int cols) {
            x_bool = true;
            y_bool = true;
            z_bool = true;
            this->rows = rows;
            this->cols = cols;
            x = ComplexMatrix(rows, cols);
            y = ComplexMatrix(rows, cols);
            z = ComplexMatrix(rows, cols);
        };  // Constructer
         Field(int rows, int cols, bool host, bool device) {
            x_bool = true;
            y_bool = true;
            z_bool = true;
            this->rows = rows;
            this->cols = cols;
            x = ComplexMatrix(rows, cols, host, device);
            y = ComplexMatrix(rows, cols, host, device);
            z = ComplexMatrix(rows, cols, host, device);
        };  // Constructer
        Field(int rows, int cols, bool x_bool_new, bool y_bool_new, bool z_bool_new) {
            x_bool = x_bool_new;
            y_bool = y_bool_new;
            z_bool = z_bool_new;
            this->rows = rows;
            this->cols = cols;
            if (x_bool) x = ComplexMatrix(rows, cols);
            if (y_bool) y = ComplexMatrix(rows, cols);
            if (z_bool) z = ComplexMatrix(rows, cols);
        }; // Constructer

        Field(int rows, int cols, bool x_bool_new, bool y_bool_new, bool z_bool_new, bool host, bool device) {
            x_bool = x_bool_new;
            y_bool = y_bool_new;
            z_bool = z_bool_new;
            this->rows = rows;
            this->cols = cols;
            if (x_bool) x = ComplexMatrix(rows, cols, host, device);
            if (y_bool) y = ComplexMatrix(rows, cols, host, device);
            if (z_bool) z = ComplexMatrix(rows, cols, host, device);
        }; // Constructer

        void free() {
            if (x_bool) x.free();
            if (y_bool) y.free();
            if (z_bool) z.free();
        }

        void setHostZero() {
            if (x_bool) x.setHostZero();
            if (y_bool) y.setHostZero();
            if (z_bool) z.setHostZero();
        }

        void setDeviceZero() {
            if (x_bool) x.setDeviceZero();
            if (y_bool) y.setDeviceZero();
            if (z_bool) z.setDeviceZero();
        }

        void toDevice() {
            if (x_bool) x.toDevice();
            if (y_bool) y.toDevice();
            if (z_bool) z.toDevice();
        }

        void toHost() {
            if (x_bool) x.toHost();
            if (y_bool) y.toHost();
            if (z_bool) z.toHost();
        }
    

};
}

#endif