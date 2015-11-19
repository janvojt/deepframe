/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include "cudaDebugHelpers.h"

#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"


void dumpDeviceInts(const char *flag, const int *dm, const int size) {
    if (LOG()->isDebugEnabled()) {
        std::cout << flag << std::endl;
        int *hdm = new int[size];
        checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(int) * size, cudaMemcpyDeviceToHost));

        for (int i = 0; i<size; i++) {
            std::cout << "Dumping device " << flag << ": " << hdm[i] << std::endl;
        }
        std::cout << "-----------------------------" << std::endl;

        delete[] hdm;
    }
}

void dumpDeviceArray(const char *flag, const data_t *dm, const int size) {
    if (LOG()->isDebugEnabled()) {
        std::cout << flag << std::endl;
        data_t *hdm = new data_t[size];
        checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(data_t) * size, cudaMemcpyDeviceToHost));

        for (int i = 0; i<size; i++) {
            std::cout << "Dumping device " << flag << ": " << hdm[i] << std::endl;
        }
        std::cout << "-----------------------------" << std::endl;

        delete[] hdm;
    }
}


void compare(const char *flag, double *dm, double *hm, const int size) {
    if (LOG()->isDebugEnabled()) {
        double *hdm = new double[size];
        checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(double) * size, cudaMemcpyDeviceToHost));

        for (int i = 0; i<size; i++) {
            if (hdm[i] == hm[i]) {
                std::cout << "Comparing " << flag << ": " << hdm[i] << " =?= " << hm[i] << std::endl;
            } else {
                std::cout << "Comparing " << flag << ": " << hdm[i] << " =?= " << hm[i] << "        !!!!!!!!!!!!!!!!!!" << std::endl;
            }
        }

        delete[] hdm;
    }
}


void isNan(const char *flag, const data_t *dm, const int size) {
    if (LOG()->isDebugEnabled()) {
        data_t *hdm = new data_t[size];
        checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(data_t) * size, cudaMemcpyDeviceToHost));

        for (int i = 0; i<size; i++) {
            if (hdm[i] != hdm[i]) {
                std::cout << "#### detected NaN for " << flag << "[" << i << "]: " << hdm[i] << std::endl;
                break;
            }
        }

        delete[] hdm;
    }
}
