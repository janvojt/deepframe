/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>


__global__
void sumVectors(double *dA, double *dB, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < elements) {
        dA[i] += dB[i];
    }
}
void k_sumVectors(double *dA, double *dB, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    sumVectors<<<bs,ts>>>(dA, dB, elements);
}

__global__
void computeOutputLocalGradient(double *actualOutput, double *expectedOutput, double *localGradient, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        double derivative = actualOutput[i] * (1.0 - actualOutput[i]);
        localGradient[i] = (actualOutput[i] - expectedOutput[i]) * derivative;
    }
}
void k_computeOutputLocalGradient(double *actualOutput, double *expectedOutput, double *localGradient, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeOutputLocalGradient<<<bs,ts>>>(actualOutput, expectedOutput, localGradient, elements);
}

__global__
void computeTotalDerivative(double learningRate, int nextNeurons,
        double *thisInput, double *nextLocalGradient,
        double *weightDiffs, int elements) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < elements) {
        int i = idx / nextNeurons;
        int j = idx % nextNeurons;
        weightDiffs[i*nextNeurons+j] = -learningRate * nextLocalGradient[j] * thisInput[i];
    }
}
void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        double learningRate, double *thisInput, double *nextLocalGradient,
        double *weightDiffs) {
    int ts = 512;
    int bs = (thisNeurons * nextNeurons + ts - 1) / ts;
    computeTotalDerivative<<<bs,ts>>>(learningRate, nextNeurons,
        thisInput, nextLocalGradient,
        weightDiffs, thisNeurons * nextNeurons);
}

__global__
void computeBiasDerivative(double learningRate, double *nextLocalGradient,
        double *biasDiffs) {
    
    int i = threadIdx.x;
    
    biasDiffs[i] = -learningRate * nextLocalGradient[i];
}
void k_computeBiasDerivative(const dim3 bs, const dim3 ts, 
        double learningRate, double *nextLocalGradient,
        double *biasDiffs) {
    computeBiasDerivative<<<bs,ts>>>(learningRate, nextLocalGradient,
        biasDiffs);
}

__global__
void computeHiddenLocalGradient(int nextNeurons,
        double *thisInput, double *weights,
        double *thisLocalGradient, double *nextLocalGradient) {
    
    int i = threadIdx.x;
    
    double derivative = thisInput[i] * (1.0 - thisInput[i]);
    
    double sumNextGradient = 0;
    for (int j = 0; j<nextNeurons; j++) {
        sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
    }
    thisLocalGradient[i] = sumNextGradient * derivative;
}
void k_computeHiddenLocalGradient(const dim3 bs, const dim3 ts, int nextNeurons,
        double *thisInput, double *weights,
        double *thisLocalGradient, double *nextLocalGradient) {
    
    computeHiddenLocalGradient<<<bs,ts>>>(nextNeurons,
        thisInput, weights,
        thisLocalGradient, nextLocalGradient);
}


__global__
void clearLayer(double *valuePtr) {
    valuePtr[threadIdx.x] = 0;
}
void k_clearLayer(const dim3 bs, const dim3 ts, double *valuePtr) {
    clearLayer<<<bs,ts>>>(valuePtr);
}


__global__
void sumArrays(double *dA, double *dB) {
    dA[threadIdx.x] += dB[threadIdx.x];
}
void k_sumArrays(const dim3 bs, const dim3 ts, double *dA, double *dB) {
    sumArrays<<<bs,ts>>>(dA, dB);
}


__global__
void computeSigmoid(double *dArray) {
	int i = threadIdx.x;
	dArray[i] = 1.0 / (1.0 + exp(-dArray[i]));
}
void k_computeSigmoid(const dim3 bs, const dim3 ts, double *dArray) {
	computeSigmoid<<<bs,ts>>>(dArray);
}