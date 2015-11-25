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
#include <cblas.h>

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CUDA_NUM_THREADS = 1024;
#else
    const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


#define CUBLAS_CHECK(condition) { \
    gpuCublasAssert(condition, __FILE__, __LINE__); \
}

inline void gpuCublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        const char *errorMsg;
        switch (code) {
            case CUBLAS_STATUS_SUCCESS:
                errorMsg = "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                errorMsg = "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                errorMsg = "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                errorMsg = "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                errorMsg = "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                errorMsg = "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                errorMsg = "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                errorMsg = "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                errorMsg = "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                errorMsg = "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                errorMsg = "Unknown cublas status";
        }
        fprintf(stderr, "CUBLAS: GPUassert: %s (%s:%d)\n", errorMsg, file, line);
        if (abort) exit(code);
    }
};

#define checkCudaErrors(ans) { gpuAssert(ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

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
 * Converts an array of floats generated from uniform distribution within
 * interval [0,1] to 1 with probability p and to 0 with probability 1-p.
 * This method is essentially converted numbers generated from uniform
 * distribution into numbers generated from binomial distribution with
 * a single trial (n=1).
 * 
 * @param p array of probabilities of one
 * @param dArray data randomly generated from uniform distribution
 * @param size data size
 */
void k_uniformToCoinFlip(data_t *p, data_t *dArray, int size);

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

void k_im2col(const data_t* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    data_t* data_col);

void k_col2im(const data_t* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, data_t* data_im);


/**
 * Delegates GEMM call to cuBLAS abstracting out derivable parameters.
 * 
 * C = α op ( B ) op ( A ) + β C
 * 
 * @param handle
 * @param TransA
 * @param TransB
 * @param M number of columns of matrix op(A) and C.
 * @param N number of rows of matrix op(B) and C.
 * @param K number of rows of matrix op(A) and columns of op(B).
 * @param alpha scalar used for multiplication of op(B) op(A).
 * @param A array, where op(A) is of dimensions K x M.
 * @param B array, where op(B) is of dimensions N x K.
 * @param beta scalar used for multiplication of C. If beta==0, C does not have to be a valid input.
 * @param C array of dimensions N x M.
 */
void k_gemm(cublasContext *handle, const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const data_t alpha, const data_t* A, const data_t* B, const data_t beta,
    data_t* C);

/**
 * Delegates AXPY call to cuBLAS.
 * Y = α X + Y
 * 
 * @param handle handle to the cuBLAS library context.
 * @param n number of elements in the vector x and y
 * @param alpha scalar used for multiplication.
 * @param x vector with n elements.
 * @param incx stride between consecutive elements of x.
 * @param y vector with n elements.
 * @param incy stride between consecutive elements of y.
 */
void k_axpy(cublasContext *handle, int n, data_t alpha, const data_t *x, int incx, data_t *y, int incy);

/**
 * Delegates SCAL call (multiplying vector with a scalar) to cuBLAS.
 * X = α X
 * 
 * @param handle handle to the cuBLAS library context.
 * @param n number of elements in the vector x and y
 * @param alpha scalar used for multiplication.
 * @param x vector with n elements.
 * @param incx stride between consecutive elements of x.
 */
void k_scal(cublasContext *handle, int n, data_t alpha, data_t *x, int incx);

/**
 * Delegates DOT call (dot product of two vectors) to cuBLAS.
 * 
 * @param handle handle to the cuBLAS library context.
 * @param n number of elements in the vector x and y
 * @param x vector with n elements.
 * @param incx stride between consecutive elements of x.
 * @param y vector with n elements.
 * @param incy stride between consecutive elements of y.
 * @param result scalar resulting from the dot product
 */
void k_dotProduct(cublasContext *handle, int n, const data_t *x, int incx, const data_t *y, int incy, data_t *result);

void k_MaxPoolForward(const int nthreads,
    const data_t* const inputs, const int channels,
    const int inputFeatureHeight, const int inputFeatureWidth, const int featureHeight,
    const int featureWidth, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    data_t* const outputs, int* mask);

void k_MaxPoolBackward(const int nthreads, const data_t* const outputDiffs,
    const int* const mask, const int featuresCount,
    const int inputFeatureHeight, const int inputFeatureWidth,
    const int featureHeight, const int featureWidth, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, data_t* const inputDiffs);

/**
 * Parallel implementation of summing an array of floats.
 * 
 * @param in input array to sum up
 * @param out temporary data store for meta-results (passing avoids memory reallocations)
 * @param n size of the input array
 * @return the sum
 */
data_t k_sumReduce(data_t *in, data_t *out, unsigned long n);

data_t k_logPlusExpReduce(data_t a, data_t *in, data_t *out, unsigned long n);

#endif	/* CUDAHELPERS_H */

