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

void compareDeviceArrays(const char *flag, const data_t *dm1, const data_t *dm2, const int size) {
    if (LOG()->isDebugEnabled()) {
        
        std::cout << "Comparing device memory (" << flag << "): " << std::endl;

        data_t *hdm1 = new data_t[size*2];
        data_t *hdm2 = hdm1 + size;
        checkCudaErrors(cudaMemcpy(hdm1, dm1, sizeof(data_t) * size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hdm2, dm2, sizeof(data_t) * size, cudaMemcpyDeviceToHost));

//        int max = size > 100 ? 100 : size;
        for (int i = 0; i<size; i+=1000) {
            data_t percent = hdm2[i] / hdm1[i] * 100;
            std::cout << "v1: " << hdm1[i] << "\t\tv2: " << hdm2[i] << "\t\tpercent: " << percent << std::endl;
        }
        std::cout << "-----------------------------" << std::endl;

        delete[] hdm1;
    }
}

void paint2DimageL(const char *flag, Layer* layer) {
    if (LOG()->isDebugEnabled()) {
        
        data_t *dm = layer->getOutputs();
        int size = layer->getOutputsCount();
        
        paint2Dimage(flag, dm, size);
    }
}

void paint2Dimage(const char* flag, data_t *data, int size) {
    if (LOG()->isDebugEnabled()) {
        
        std::cout << flag << std::endl;
        
        int x = sqrt(size);
        int y = x;
        
        data_t *hdm = new data_t[size];
        checkCudaErrors(cudaMemcpy(hdm, data, sizeof(data_t) * size, cudaMemcpyDeviceToHost));
        
        for (int i = 0; i<x; i++) {
            char sep = ' ';
            std::cout << sep;
            for (int j = 0; j<y; j++) {
                data_t d = hdm[i*x+j];
                if (d<.5) {
                    std::cout << " ";
                } else if (d<.9) {
                    std::cout << "<";
                } else if (d<=1.) {
                    std::cout << "#";
                } else {
                    std::cout << "&";
                }
            }
            std::cout << std::endl;
        }
        
        std::cout << "-----------------------------" << std::endl;

        delete[] hdm;
    }
}

void compare2Dimages(const char* flag, data_t *data1, data_t *data2, int size) {
    if (LOG()->isDebugEnabled()) {
        
        std::cout << flag << std::endl;
        
        int x = sqrt(size);
        int y = x;
        
        data_t *hdm1 = new data_t[size * 2];
        data_t *hdm2 = hdm1 + size;
        checkCudaErrors(cudaMemcpy(hdm1, data1, sizeof(data_t) * size, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(hdm2, data2, sizeof(data_t) * size, cudaMemcpyDeviceToHost));
        
        for (int i = 0; i<x; i++) {
            char sep = ' ';
            std::cout << sep;
            for (int j = 0; j<y; j++) {
                data_t d = hdm1[i*x+j];
                if (d<.5) {
                    std::cout << " ";
                } else if (d<.9) {
                    std::cout << "<";
                } else if (d<=1.) {
                    std::cout << "#";
                } else {
                    std::cout << "&";
                }
            }
            
            std::cout << "        ";
            
            for (int j = 0; j<y; j++) {
                data_t d = hdm2[i*x+j];
                if (d<.5) {
                    std::cout << " ";
                } else if (d<.9) {
                    std::cout << "<";
                } else if (d<=1.) {
                    std::cout << "#";
                } else {
                    std::cout << "&";
                }
            }
            std::cout << std::endl;
        }
        
        std::cout << "-------------------------------------------" << std::endl;

        delete[] hdm1;
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
