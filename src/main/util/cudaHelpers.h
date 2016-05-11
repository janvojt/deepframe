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

// Define function for verifying cuBLAS return status codes.
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

// Defines function for verifying cuRAND return status codes.
#define CURAND_CHECK(condition) { \
gpuCurandAssert(condition, __FILE__, __LINE__); \
}

inline void gpuCurandAssert(curandStatus_t code, const char *file, int line, bool abort = true) {
    if (code != CURAND_STATUS_SUCCESS) {
        const char *errorMsg;
        switch (code) {
            case CURAND_STATUS_SUCCESS:
                errorMsg = "No errors.";
            case CURAND_STATUS_VERSION_MISMATCH:
                errorMsg = "Header file and linked library version do not match.";
            case CURAND_STATUS_NOT_INITIALIZED:
                errorMsg = "Generator not initialized.";
            case CURAND_STATUS_ALLOCATION_FAILED:
                errorMsg = "Memory allocation failed.";
            case CURAND_STATUS_TYPE_ERROR:
                errorMsg = "Generator is wrong type.";
            case CURAND_STATUS_OUT_OF_RANGE:
                errorMsg = "Argument out of range.";
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                errorMsg = "Length requested is not a multiple of dimension.";
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                errorMsg = "GPU does not have double precision required by MRG32k3a.";
            case CURAND_STATUS_LAUNCH_FAILURE:
                errorMsg = "Kernel launch failure.";
            case CURAND_STATUS_PREEXISTING_FAILURE:
                errorMsg = "Preexisting failure on library entry";
            case CURAND_STATUS_INITIALIZATION_FAILED:
                errorMsg = "Initialization of CUDA failed.";
            case CURAND_STATUS_ARCH_MISMATCH:
                errorMsg = "Architecture mismatch, GPU does not support requested feature.";
            case CURAND_STATUS_INTERNAL_ERROR:
                errorMsg = "Internal library error.";
            default:
                errorMsg = "Unknown cuRand status";
        }
        fprintf(stderr, "CUBLAS: GPUassert: %s (%s:%d, error code:%d)\n", errorMsg, file, line, code);
        if (abort) exit(code);
    }
};

// Defines function for verifying the CUDA return status codes.
#define checkCudaErrors(ans) { gpuAssert(ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Defines function for CUDA grid stride looping.
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

/**
 * Computes vector sum A = A + B.
 * 
 * @param dA pointer to vector A stored on device
 * @param dB pointer to vector B stored on device
 * @param elements number of dimensions of both vectors
 */
void k_sumVectors(data_t *dA, data_t *dB, int elements);

/**
 * Divides every array element by given divisor.
 * 
 * @param dA array to divide
 * @param divisor
 * @param elements size of the array
 */
void k_divideVector(data_t *dA, int divisor, int elements);

/**
 * Computes the gradients for weights in the perceptron layer
 * based on the expected output. This kernel is only used in
 * the very last layer of the network, where the expected
 * output is known.
 * 
 * @param actualOutput actual output of the perceptron layer
 * @param expectedOutput expected output from observed example
 * @param localGradient computed local gradients
 * @param elements number of neurons in the layer (output size)
 */
void k_computeOutputLocalGradient(data_t *actualOutput, data_t *expectedOutput, data_t *localGradient, int elements);

/**
 * Computes the total derivative for weights coming into the next layer.
 * 
 * @param thisNeurons number of neurons in the current layer (output neurons)
 * @param nextNeurons number of neurons in the next layer (output neurons)
 * @param learningRate learning rate used to normalize the derivatives
 * @param thisInput pointer to the activation values for neurons in the current layer
 * @param nextLocalGradient pointer to the local gradient values for next layer
 * @param weightDiffs the computed total derivatives for weights between this and next layer
 */
void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        data_t learningRate, data_t *thisInput, data_t *nextLocalGradient,
        data_t *weightDiffs);

/**
 * Compute the derivatives for biases.
 * 
 * @param learningRate learning rate used to normalize the bias
 * @param nextLocalGradient local gradient values in the next layer
 * @param biasDiffs the computed derivatives for biases
 * @param elements number of bias values (neurons)
 */
void k_computeBiasDerivative(
        data_t learningRate, data_t *nextLocalGradient,
        data_t *biasDiffs, int elements);

/**
 * Computes the local gradients for current layer.
 * 
 * @param thisNeurons number of neurons in the current layer
 * @param nextNeurons number of neurons in the next layer
 * @param thisInput pointer to the activations in current layer
 * @param weights weights between the current and the next layer
 * @param thisLocalGradient the computed local gradient for the current layer
 * @param nextLocalGradient the local gradient for the next layer
 */
void k_computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        data_t *thisInput, data_t *weights,
        data_t *thisLocalGradient, data_t *nextLocalGradient);

/**
 * Computes the sigmoid function on given input array and stores the result in
 * the output array. The input and output can actually point to the same memory.
 * 
 * @param inArray input array
 * @param outArray output array with the results
 * @param elements size of arrays
 */
void k_computeSigmoid(data_t *inArray, data_t *outArray, int elements);

/**
 * Assumes array of double values between 0 and 1 in dArray
 * and spreads (stretches) these values to the given interval.
 * 
 * @param min minimum value of the new interval
 * @param max maximum values of the new interval
 * @param dArray pointer to the array with values stored on device
 * @param size size of the array
 */
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
 * Converts floating point values from interval (0,1) into
 * binary states of 0 and 1. The conversion is simply done
 * by rounding the decimal number to an integer.
 * 
 * @param p pointer to an array with decimal values stored on device
 * @param size size of the array
 */
void k_flattenToCoinFlip(data_t *p, int size);

/**
 * Delegates random number generation to appropriate cuRAND call of correct
 * data type. The numbers are generated from uniform distribution from the
 * interval between 0.0 (excluding) and 1.0 (including).
 * 
 * @param generator
 * @param outputPtr
 * @param num
 * @return 
 */
curandStatus_t k_generateUniform(curandGenerator_t generator,
        data_t *outputPtr,
        size_t num);

/**
 * Delegates random number generation to appropriate cuRAND call of correct
 * data type. The numbers are generated from Gaussian distribution with the
 * given mean and standard deviation.
 * 
 * @param generator
 * @param outputPtr
 * @param num
 * @param mean
 * @param stddev
 * @return 
 */
curandStatus_t k_generateNormal(curandGenerator_t generator,
        data_t *outputPtr,
        size_t num,
        data_t mean,
        data_t stddev);

/**
 * Converts images to column buffers, which allow usage
 * of common algebraic operations to compute the convolutions.
 * 
 * @param data_im image data
 * @param channels channels in the image (RGB, etc.)
 * @param height height of the image
 * @param width width of the image
 * @param kernel_h height of the convolution kernel
 * @param kernel_w width of the convolution kernel
 * @param pad_h padded height
 * @param pad_w padded width
 * @param stride_h vertical stride
 * @param stride_w horizontal stride
 * @param data_col column buffer
 */
void k_im2col(const data_t* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    data_t* data_col);

/**
 * Converts the column buffer back to the image.
 * 
 * @param data_col column buffer
 * @param channels number of channels in the image (RGB, etc.)
 * @param height height of the image
 * @param width width of the image
 * @param patch_h height of the convolution kernel
 * @param patch_w width of the convolution kernel
 * @param pad_h padded height
 * @param pad_w padded width
 * @param stride_h vertical stride
 * @param stride_w horizontal stride
 * @param data_im pointer to the generated image
 */
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

/**
 * Performs max pooling (subsampling) on the input feature maps.
 * 
 * @param outputsCount pooling size
 * @param inputs pointer to the data in feature maps
 * @param channels number of channels (RGB, etc.)
 * @param inputFeatureHeight height of input feature maps
 * @param inputFeatureWidth width of input feature maps
 * @param featureHeight height of the pooled features
 * @param featureWidth width of the pooled features
 * @param kernel_h subsampling kernel width
 * @param kernel_w subsampling kernel height
 * @param stride_h vertical stride
 * @param stride_w horizontal stride
 * @param pad_h padded height
 * @param pad_w padded width
 * @param outputs pointer to the produced subsampled feature data
 * @param mask mask for caching the information which neuron activations are passed forward
 */
void k_MaxPoolForward(const int outputsCount,
    const data_t* const inputs, const int channels,
    const int inputFeatureHeight, const int inputFeatureWidth, const int featureHeight,
    const int featureWidth, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    data_t* const outputs, int* mask);

/**
 * Propagates the gradients back through the subsampling layer.
 * 
 * @param inputsCount number of neurons in input feature maps
 * @param outputDiffs the gradients for output of the current subsampling layer
 * @param mask mask with cached information about which neuron activation was passed in the forward phase
 * @param featuresCount number of input feature maps
 * @param inputFeatureHeight height of input feature maps
 * @param inputFeatureWidth width of input feature maps
 * @param featureHeight height of subsampled feature maps
 * @param featureWidth width of subsampled feature maps
 * @param kernel_h subsampling kernel height
 * @param kernel_w subsampling kernel width
 * @param stride_h vertical stride
 * @param stride_w horizontal stride
 * @param pad_h padded height
 * @param pad_w padded width
 * @param inputDiffs computed gradients for input neurons
 */
void k_MaxPoolBackward(const int inputsCount, const data_t* const outputDiffs,
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

/**
 * Parallel implementation summing log(a + e^x).
 * 
 * @param a addition inside log
 * @param in input array (x)
 * @param out temporary array for partial results
 * @param n input size
 * @return the sum of logs
 */
data_t k_logPlusExpReduce(data_t a, data_t *in, data_t *out, unsigned long n);

/**
 * Computes cross-entropy error v*log(sig(pv)) + (1-v)*log(1-sig(pv)).
 * 
 * @param visibles states of visible neurons
 * @param potentials potentials for visible neurons
 * @param temp temporary store for partial results
 * @param n number of neurons/potentials
 * @return cross-entropy
 */
data_t k_crossEntropyReduce(data_t *visibles, data_t *potentials, data_t *temp, unsigned long n);

#endif	/* CUDAHELPERS_H */

