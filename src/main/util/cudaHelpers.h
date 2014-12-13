/* 
 * File:   cudaHelpers.h
 * Author: janvojt
 *
 * Created on November 29, 2014, 11:49 PM
 */

#ifndef CUDAHELPERS_H
#define	CUDAHELPERS_H

#include <cstdlib>
#include <iostream>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s(%d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Computes matrix sum A = A + B.
void k_sumVectors(double *dA, double *dB, int elements);

void k_computeOutputLocalGradient(double *actualOutput, double *expectedOutput, double *localGradient, int elements);

void k_computeTotalDerivative(const dim3 bs, const dim3 ts, 
        double learningRate, int nextNeurons,
        double *thisInput, double *nextLocalGradient,
        double *weightDiffs);

void k_computeBiasDerivative(const dim3 bs, const dim3 ts, 
        double learningRate, double *nextLocalGradient,
        double *biasDiffs);

void k_computeHiddenLocalGradient(const dim3 bs, const dim3 ts, 
        int nextNeurons,
        double *thisInput, double *weights,
        double *thisLocalGradient, double *nextLocalGradient);

// Sets all the values in an array to zeros.
void k_clearLayer(const dim3 bs, const dim3 ts, double *valuePtr);

// Compute A = A + B.
void k_sumArrays(const dim3 bs, const dim3 ts, double *dA, double *dB);

// Compute the sigmoid function on device array.
void k_computeSigmoid(const dim3 bs, const dim3 ts, double *dArray);

#endif	/* CUDAHELPERS_H */

