/* 
 * File:   cudaHelpers.h
 * Author: janvojt
 *
 * Created on November 29, 2014, 11:49 PM
 */

#ifndef CUDAHELPERS_H
#define	CUDAHELPERS_H

#include "../common.h"

#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <curand.h>
#include <cublas_v2.h>

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Computes matrix sum A = A + B.
void k_sumVectors(data_t *dA, data_t *dB, int elements);

/**
 * Divides every array element by given divisor.
 * 
 * @param dA array to divide
 * @param divisor
 * @param elements size of the array
 */
void k_divideVector(data_t *dA, int divisor, int elements);

void k_computeOutputLocalGradient(data_t *actualOutput, data_t *expectedOutput, data_t *localGradient, int elements);

void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        data_t learningRate, data_t *thisInput, data_t *nextLocalGradient,
        data_t *weightDiffs);

void k_computeBiasDerivative(
        data_t learningRate, data_t *nextLocalGradient,
        data_t *biasDiffs, int elements);

void k_computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        data_t *thisInput, data_t *weights,
        data_t *thisLocalGradient, data_t *nextLocalGradient);

// Compute the sigmoid function on device array.
void k_computeSigmoid(data_t *dArray, int elements);

// Assumes array of double values between 0 and 1 in dArray and 
// spreads this to given interval.
void k_spreadInterval(data_t min, data_t max, data_t *dArray, int size);

/**
 * Delegates GEMM operation to appropriate cuBLAS call of correct data type.
 * 
 * @param handle
 * @param transa
 * @param transb
 * @param m
 * @param n
 * @param k
 * @param alpha
 * @param A
 * @param lda
 * @param B
 * @param ldb
 * @param beta
 * @param C
 * @param ldc
 * @return 
 */
cublasStatus_t k_gemm(cublasContext *handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const data_t *alpha, /* host or device pointer */
        const data_t *A,
        int lda,
        const data_t *B,
        int ldb,
        const data_t *beta, /* host or device pointer */
        data_t *C,
        int ldc);
/**
 * Delegates random number generation to appropriate cuRAND call of correct
 * data type.
 * 
 * @param generator
 * @param outputPtr
 * @param num
 * @return 
 */
curandStatus_t k_generateUniform(curandGenerator_t generator,
        data_t *outputPtr,
        size_t num);


#endif	/* CUDAHELPERS_H */

