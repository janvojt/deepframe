/* 
 * File:   cudaHelpers.h
 * Author: janvojt
 *
 * Created on November 29, 2014, 11:49 PM
 */

#ifndef CUDAHELPERS_H
#define	CUDAHELPERS_H

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
template <typename dType>
void k_sumVectors(dType *dA, dType *dB, int elements);

/**
 * Divides every array element by given divisor.
 * 
 * @param dA array to divide
 * @param divisor
 * @param elements size of the array
 */
template <typename dType>
void k_divideVector(dType *dA, int divisor, int elements);

template <typename dType>
void k_computeOutputLocalGradient(dType *actualOutput, dType *expectedOutput, dType *localGradient, int elements);

template <typename dType>
void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        dType learningRate, dType *thisInput, dType *nextLocalGradient,
        dType *weightDiffs);

template <typename dType>
void k_computeBiasDerivative(
        dType learningRate, dType *nextLocalGradient,
        dType *biasDiffs, int elements);

template <typename dType>
void k_computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        dType *thisInput, dType *weights,
        dType *thisLocalGradient, dType *nextLocalGradient);

// Compute the sigmoid function on device array.
template <typename dType>
void k_computeSigmoid(dType *dArray, int elements);

// Assumes array of double values between 0 and 1 in dArray and 
// spreads this to given interval.
template <typename dType>
void k_spreadInterval(dType min, dType max, dType *dArray, int size);

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
template <typename dType>
cublasStatus_t k_gemm(cublasContext *handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const dType *alpha, /* host or device pointer */
        const dType *A,
        int lda,
        const dType *B,
        int ldb,
        const dType *beta, /* host or device pointer */
        dType *C,
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
template <typename dType>
curandStatus_t k_generateUniform(curandGenerator_t generator,
        dType *outputPtr,
        size_t num);


#endif	/* CUDAHELPERS_H */

