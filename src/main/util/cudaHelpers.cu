/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include "cudaDebugHelpers.h"

#include <cuda_runtime.h>
#include <cuda.h>


__global__
void sumVectors(data_t *dA, data_t *dB, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dA[i] += dB[i];
    }
}
void k_sumVectors(data_t *dA, data_t *dB, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    sumVectors<<<bs,ts>>>(dA, dB, elements);
}


__global__
void divideVector(data_t *dA, int divisor, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dA[i] /= divisor;
    }
}
void k_divideVector(data_t *dA, int divisor, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    divideVector<<<bs,ts>>>(dA, divisor, elements);
}


__global__
void computeOutputLocalGradient(data_t *actualOutput, data_t *expectedOutput, data_t *localGradient, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        data_t derivative = actualOutput[i] * (1.0 - actualOutput[i]);
        localGradient[i] = (actualOutput[i] - expectedOutput[i]) * derivative;
    }
}
void k_computeOutputLocalGradient(data_t *actualOutput, data_t *expectedOutput, data_t *localGradient, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeOutputLocalGradient<<<bs,ts>>>(actualOutput, expectedOutput, localGradient, elements);
}


__global__
void computeTotalDerivative(data_t learningRate, int nextNeurons,
        data_t *thisInput, data_t *nextLocalGradient,
        data_t *weightDiffs, int elements) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < elements) {
        int i = idx / nextNeurons;
        int j = idx % nextNeurons;
        weightDiffs[i*nextNeurons+j] = -learningRate * nextLocalGradient[j] * thisInput[i];
    }
}
void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        data_t learningRate, data_t *thisInput, data_t *nextLocalGradient,
        data_t *weightDiffs) {
    int ts = 512;
    int bs = (thisNeurons * nextNeurons + ts - 1) / ts;
    computeTotalDerivative<<<bs,ts>>>(learningRate, nextNeurons,
        thisInput, nextLocalGradient,
        weightDiffs, thisNeurons * nextNeurons);
}


__global__
void computeBiasDerivative(data_t learningRate, data_t *nextLocalGradient,
        data_t *biasDiffs, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        biasDiffs[i] = -learningRate * nextLocalGradient[i];
    }
}
void k_computeBiasDerivative(
        data_t learningRate, data_t *nextLocalGradient,
        data_t *biasDiffs, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeBiasDerivative<<<bs,ts>>>(learningRate, nextLocalGradient,
        biasDiffs, elements);
}


__global__
void computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        data_t *thisInput, data_t *weights,
        data_t *thisLocalGradient, data_t *nextLocalGradient) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < thisNeurons) {
        data_t derivative = thisInput[i] * (1.0 - thisInput[i]);

        data_t sumNextGradient = 0;
        for (int j = 0; j<nextNeurons; j++) {
            sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
        }
        thisLocalGradient[i] = sumNextGradient * derivative;
    }
}
void k_computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        data_t *thisInput, data_t *weights,
        data_t *thisLocalGradient, data_t *nextLocalGradient) {
    
    int ts = 512;
    int bs = (thisNeurons + ts - 1) / ts;
    computeHiddenLocalGradient<<<bs,ts>>>(
        thisNeurons, nextNeurons,
        thisInput, weights,
        thisLocalGradient, nextLocalGradient);
}


__global__
void computeSigmoid(data_t *dArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dArray[i] = 1.0 / (1.0 + exp(-dArray[i]));
    }
}
void k_computeSigmoid(data_t *dArray, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
	computeSigmoid<<<bs,ts>>>(dArray, elements);
}


__global__
void spreadInterval(data_t min, data_t max, data_t *dArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dArray[i] = (dArray[i] * (max - min)) + min;
    }
}
void k_spreadInterval(data_t min, data_t max, data_t *dArray, int size) {
    int ts = 512;
    int bs = (size + ts - 1) / ts;
    spreadInterval<<<bs,ts>>>(min, max, dArray, size);
}


/** Delegates GEMM call to cuBLAS for single precision. */
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
        float *C,
        int ldc) {
    
#ifdef USE_64BIT_PRECISION
    return cublasDgemm(handle,
            transa, transb,
            m, n, k,
            alpha, A, lda,
            B, ldb,
            beta, C, ldc);
#else
    return cublasSgemm(handle,
            transa, transb,
            m, n, k,
            alpha, A, lda,
            B, ldb,
            beta, C, ldc);
#endif
}

curandStatus_t k_generateUniform(curandGenerator_t generator,
        data_t *outputPtr,
        size_t num) {
    
#ifdef USE_64BIT_PRECISION
    return curandGenerateUniformDouble(generator, outputPtr, num);
#else
    return curandGenerateUniform(generator, outputPtr, num);
#endif
}
