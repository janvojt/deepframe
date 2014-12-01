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


void dumpDeviceArray(char flag, double *dm, int size) {
    std::cout << flag << std::endl;
    double *hdm = new double[size];
    checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(double) * size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i<size; i++) {
        std::cout << "Dumping device " << flag << ": " << hdm[i] << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
    
    delete[] hdm;
}


void compare(char flag, double *dm, double *hm, int size) {
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


void printArray(char flag, double *dm, int size) {
    double *hdm = new double[size];
    checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(double) * size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i<size; i++) {
        std::cout << "Printing device memory " << flag << ": " << hdm[i] << std::endl;
    }
    
    delete[] hdm;
}
