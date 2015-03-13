/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include "cudaDebugHelpers.h"

#include <cuda_runtime.h>
#include <cuda.h>


template <typename dType>
__global__
void sumVectors(dType *dA, dType *dB, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dA[i] += dB[i];
    }
}
template <typename dType>
void k_sumVectors(dType *dA, dType *dB, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    sumVectors<<<bs,ts>>>(dA, dB, elements);
}
template __global__ void sumVectors<float>(float*, float*, int);
template __global__ void sumVectors<double>(double*, double*, int);
template void k_sumVectors<float>(float*, float*, int);
template void k_sumVectors<double>(double*, double*, int);


template <typename dType>
__global__
void divideVector(dType *dA, int divisor, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dA[i] /= divisor;
    }
}
template <typename dType>
void k_divideVector(dType *dA, int divisor, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    divideVector<<<bs,ts>>>(dA, divisor, elements);
}
template __global__ void divideVector<float>(float*, int, int);
template __global__ void divideVector<double>(double*, int, int);
template void k_divideVector<float>(float*, int, int);
template void k_divideVector<double>(double*, int, int);


template <typename dType>
__global__
void computeOutputLocalGradient(dType *actualOutput, dType *expectedOutput, dType *localGradient, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dType derivative = actualOutput[i] * (1.0 - actualOutput[i]);
        localGradient[i] = (actualOutput[i] - expectedOutput[i]) * derivative;
    }
}
template <typename dType>
void k_computeOutputLocalGradient(dType *actualOutput, dType *expectedOutput, dType *localGradient, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeOutputLocalGradient<<<bs,ts>>>(actualOutput, expectedOutput, localGradient, elements);
}
template __global__ void computeOutputLocalGradient<float>(float*, float*, float*, int);
template __global__ void computeOutputLocalGradient<double>(double*, double*, double*, int);
template void k_computeOutputLocalGradient<float>(float*, float*, float*, int);
template void k_computeOutputLocalGradient<double>(double*, double*, double*, int);


template <typename dType>
__global__
void computeTotalDerivative(dType learningRate, int nextNeurons,
        dType *thisInput, dType *nextLocalGradient,
        dType *weightDiffs, int elements) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < elements) {
        int i = idx / nextNeurons;
        int j = idx % nextNeurons;
        weightDiffs[i*nextNeurons+j] = -learningRate * nextLocalGradient[j] * thisInput[i];
    }
}
template <typename dType>
void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        dType learningRate, dType *thisInput, dType *nextLocalGradient,
        dType *weightDiffs) {
    int ts = 512;
    int bs = (thisNeurons * nextNeurons + ts - 1) / ts;
    computeTotalDerivative<<<bs,ts>>>(learningRate, nextNeurons,
        thisInput, nextLocalGradient,
        weightDiffs, thisNeurons * nextNeurons);
}
template __global__ void computeTotalDerivative<float>(float, int, float*, float*, float*, int);
template __global__ void computeTotalDerivative<double>(double, int, double*, double*, double*, int);
template void k_computeTotalDerivative<float>(int, int, float, float*, float*, float*);
template void k_computeTotalDerivative<double>(int, int, double, double*, double*, double*);


template <typename dType>
__global__
void computeBiasDerivative(dType learningRate, dType *nextLocalGradient,
        dType *biasDiffs, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        biasDiffs[i] = -learningRate * nextLocalGradient[i];
    }
}
template <typename dType>
void k_computeBiasDerivative(
        dType learningRate, dType *nextLocalGradient,
        dType *biasDiffs, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeBiasDerivative<<<bs,ts>>>(learningRate, nextLocalGradient,
        biasDiffs, elements);
}
template __global__ void computeBiasDerivative<float>(float, float*, float*, int);
template __global__ void computeBiasDerivative<double>(double, double*, double*, int);
template void k_computeBiasDerivative<float>(float, float*, float*, int);
template void k_computeBiasDerivative<double>(double, double*, double*, int);


template <typename dType>
__global__
void computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        dType *thisInput, dType *weights,
        dType *thisLocalGradient, dType *nextLocalGradient) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < thisNeurons) {
        dType derivative = thisInput[i] * (1.0 - thisInput[i]);

        dType sumNextGradient = 0;
        for (int j = 0; j<nextNeurons; j++) {
            sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
        }
        thisLocalGradient[i] = sumNextGradient * derivative;
    }
}
template <typename dType>
void k_computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        dType *thisInput, dType *weights,
        dType *thisLocalGradient, dType *nextLocalGradient) {
    
    int ts = 512;
    int bs = (thisNeurons + ts - 1) / ts;
    computeHiddenLocalGradient<<<bs,ts>>>(
        thisNeurons, nextNeurons,
        thisInput, weights,
        thisLocalGradient, nextLocalGradient);
}
template __global__ void computeHiddenLocalGradient<float>(int, int, float*, float*, float*, float*);
template __global__ void computeHiddenLocalGradient<double>(int, int, double*, double*, double*, double*);
template void k_computeHiddenLocalGradient<float>(int, int, float*, float*, float*, float*);
template void k_computeHiddenLocalGradient<double>(int, int, double*, double*, double*, double*);


template <typename dType>
__global__
void computeSigmoid(dType *dArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dArray[i] = 1.0 / (1.0 + exp(-dArray[i]));
    }
}
template <typename dType>
void k_computeSigmoid(dType *dArray, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
	computeSigmoid<<<bs,ts>>>(dArray, elements);
}
template __global__ void computeSigmoid<float>(float*, int);
template __global__ void computeSigmoid<double>(double*, int);
template void k_computeSigmoid<float>(float*, int);
template void k_computeSigmoid<double>(double*, int);


template <typename dType>
__global__
void spreadInterval(dType min, dType max, dType *dArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dArray[i] = (dArray[i] * (max - min)) + min;
    }
}
template <typename dType>
void k_spreadInterval(dType min, dType max, dType *dArray, int size) {
    int ts = 512;
    int bs = (size + ts - 1) / ts;
    spreadInterval<<<bs,ts>>>(min, max, dArray, size);
}
template __global__ void spreadInterval<float>(float, float, float*, int);
template __global__ void spreadInterval<double>(double, double, double*, int);
template void k_spreadInterval<float>(float, float, float*, int);
template void k_spreadInterval<double>(double, double, double*, int);


/** Delegates GEMM call to cuBLAS for single precision. */
template <>
cublasStatus_t k_gemm<float>(cublasContext *handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const float *alpha, /* host or device pointer */
        const float *A,
        int lda,
        const float *B,
        int ldb,
        const float *beta, /* host or device pointer */
        float *C,
        int ldc) {
    return cublasSgemm(handle,
            transa, transb,
            m, n, k,
            alpha, A, lda,
            B, ldb,
            beta, C, ldc);
}

/** Delegates GEMM call to cuBLAS for double precision. */
template <>
cublasStatus_t k_gemm<double>(cublasContext *handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const double *alpha, /* host or device pointer */
        const double *A,
        int lda,
        const double *B,
        int ldb,
        const double *beta, /* host or device pointer */
        double *C,
        int ldc) {
    return cublasDgemm(handle,
            transa, transb,
            m, n, k,
            alpha, A, lda,
            B, ldb,
            beta, C, ldc);
}

template <>
curandStatus_t k_generateUniform<float>(curandGenerator_t generator,
        float *outputPtr,
        size_t num) {
    return curandGenerateUniform(generator, outputPtr, num);
}

template <>
curandStatus_t k_generateUniform<double>(curandGenerator_t generator,
        double *outputPtr,
        size_t num) {
    return curandGenerateUniformDouble(generator, outputPtr, num);
}