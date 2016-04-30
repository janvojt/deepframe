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

void paint2Dimage(const char *flag, Layer* layer) {
    if (LOG()->isDebugEnabled()) {
        std::cout << flag << std::endl;
        
        data_t *dm = layer->getOutputs();
        int size = layer->getOutputsCount();
        int x = sqrt(size);
        int y = x;
        
        data_t *hdm = new data_t[size];
        checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(data_t) * size, cudaMemcpyDeviceToHost));

        for (int i = 0; i<x; i++) {
            char sep = ' ';
            std::cout << sep;
            for (int j = 0; j<y; j++) {
                data_t d = hdm[i*x+j];
                if (d<.5) {
                    std::cout << " ";
                } else if (d<.9) {
                    std::cout << "0";
                } else {
                    std::cout << "#";
                }
            }
            std::cout << std::endl;
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
