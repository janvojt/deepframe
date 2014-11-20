/* 
 * File:   GpuNetwork.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "GpuNetwork.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"
#include "GpuConfiguration.h"

#include <cstring>
#include <string>
#include <stdlib.h>
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

// Sets all the values in an array to zeros.
__global__
void clearLayer(double *valuePtr) {
    valuePtr[threadIdx.x] = 0;
}

// Compute A = A + B.
__global__
void sumArrays(double *dA, double *dB) {
    dA[threadIdx.x] += dB[threadIdx.x];
}

// Compute the sigmoid function on device array.
__global__
void computeSigmoid(double *dArray) {
	int i = threadIdx.x;
	dArray[i] = 1.0 / (1.0 + exp(-dArray[i]));
}

// TODO temporary function for debugging purposes
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
// TODO temporary function for debugging purposes
void printArray(char flag, double *dm, int size) {
    double *hdm = new double[size];
    checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(double) * size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i<size; i++) {
        std::cout << "Printing device memory " << flag << ": " << hdm[i] << std::endl;
    }
    
    delete[] hdm;
}

GpuNetwork::GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf) : Network(netConf) {
    cublasCreate(&cublasHandle);
    this->gpuConf = gpuConf;
    initWeights();
    initInputs();
    initBias();
}

GpuNetwork::GpuNetwork(const GpuNetwork& orig) : Network(orig) {
}

GpuNetwork::~GpuNetwork() {
    cublasDestroy(cublasHandle);
    cudaFree(dWeights);
    cudaFree(dBias);
    delete[] weightsUpToLayerCache;
    delete[] neuronsUpToLayerCache;
    delete[] weights;
    delete[] inputs;
    delete[] bias;
}

void GpuNetwork::randomizeDoublesOnGpu(double **hMemory, double **dMemory, int size) {

    int memSize = sizeof(double) * size;
    checkCudaErrors(cudaMalloc(dMemory, memSize));
    
    // Initialize random values on GPU device memory.
    curandGenerateUniformDouble(*gpuConf->getRandGen(), *dMemory, size);
    
    // Copy to host memory.
    *hMemory = new double[size];
    checkCudaErrors(cudaMemcpy(*hMemory, *dMemory, memSize, cudaMemcpyDeviceToHost));
}

void GpuNetwork::initWeights() {
    int noWeights = 0;
    int pLayer = 1; // neurons in previous layer
    weightsUpToLayerCache = new int[noLayers+1];
    weightsUpToLayerCache[0] = noWeights;
    for (int i = 0; i<noLayers; i++) {
        int tLayer = this->conf->getNeurons(i);
        noWeights += pLayer * tLayer;
        weightsUpToLayerCache[i+1] = noWeights;
        pLayer = tLayer;
    }
    
    int memSize = sizeof(double) * noWeights;
    checkCudaErrors(cudaMalloc(&dWeights, memSize));
    
    // TODO use GPU random init,
    // init on CPU is done for debugging purposes so we can compare results.
//    randomizeDoublesOnGpu(&weights, &dWeights, noWeights);
    weights = new double[noWeights];
    for (int i = 0; i < noWeights; i++) {
        weights[i] = (double) (rand()) / (RAND_MAX / 2) - 1;
    }
    checkCudaErrors(cudaMemcpy(dWeights, weights, memSize, cudaMemcpyHostToDevice));
}

void GpuNetwork::initInputs() {
    int noNeurons = 0;
    neuronsUpToLayerCache = new int[noLayers+1];
    neuronsUpToLayerCache[0] = noNeurons;
    for (int i = 0; i<noLayers; i++) {
        noNeurons += conf->getNeurons(i);
        neuronsUpToLayerCache[i+1] = noNeurons;
    }
    this->noNeurons = noNeurons;
    
    // allocate host memory
    inputs = new double[noNeurons];
    
    // allocate device memory
    int memSize = sizeof(double) * noNeurons;
    checkCudaErrors(cudaMalloc(&dInputs, memSize));
}

void GpuNetwork::initBias() {
    if (conf->getBias()) {
    
        int memSize = sizeof(double) * noNeurons;
        checkCudaErrors(cudaMalloc(&dBias, memSize));
        
        // TODO uncomment to initialize on GPU
        // Initialize bias.
//        randomizeDoublesOnGpu(&bias, &dBias, noNeurons);
        // Randomly initialize bias between -1 and 1.
        bias = new double[noNeurons];
        for (int i = 0; i < noNeurons; i++) {
            bias[i] = (double) (rand()) / (RAND_MAX / 2) - 1;
        }
        checkCudaErrors(cudaMemcpy(dBias, bias, memSize, cudaMemcpyHostToDevice));
        
    } else {
        bias = NULL;
    }
}

void GpuNetwork::run() {
    // number of neurons in so far processed layers
    double *dWeightsPtr = dWeights + getInputNeurons();
    double *dInputsPtr = dInputs;
    double *dBiasPtr = dBias;
    
    // copy weights and bias from host to device
    int wMemSize = sizeof(double) * getWeightsOffset(noLayers);
    int iMemSize = sizeof(double) * getInputOffset(noLayers);
    checkCudaErrors(cudaMemcpy(dInputs, inputs, iMemSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dWeights, weights, wMemSize, cudaMemcpyHostToDevice));
    if (conf->getBias()) checkCudaErrors(cudaMemcpy(dBias, bias, iMemSize, cudaMemcpyHostToDevice));
    
    // for every layer
    for (int l = 0; l<noLayers-1; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        clearLayer<<<1,nNextLayer>>>(dInputsPtr + nThisLayer);
        
        //note cuBLAS is column primary!
        //need to transpose the order
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasDgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                1, nNextLayer, nThisLayer,
                &alpha, dInputsPtr, 1,
                dWeightsPtr, nNextLayer,
                &beta, dInputsPtr+nThisLayer, 1);

        if (conf->getBias()) {
            sumArrays<<<1,nThisLayer>>>(dInputsPtr + nThisLayer, dBiasPtr + nThisLayer);
            dBiasPtr += nThisLayer;
        }
        
//        printArray('a', dInputsPtr + nThisLayer, nNextLayer);
        computeSigmoid<<<1,nNextLayer>>>(dInputsPtr + nThisLayer);
//        printArray('b', dInputsPtr + nThisLayer, nNextLayer);
	
        dWeightsPtr += nThisLayer * nNextLayer;
        dInputsPtr += nThisLayer;
    }
    
    // copy all weights and bias back to host
    checkCudaErrors(cudaMemcpy(inputs, dInputs, iMemSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(weights, dWeights, wMemSize, cudaMemcpyDeviceToHost));
    if (conf->getBias()) checkCudaErrors(cudaMemcpy(bias, dBias, iMemSize, cudaMemcpyDeviceToHost));
    
//    compare('w', dWeights, weights, getWeightsOffset(noLayers));
//    if (conf->getBias()) compare('b', dBias, bias, getInputOffset(noLayers));
//    compare('i', dInputs, inputs, getInputOffset(noLayers));
}

void GpuNetwork::setInput(double* input) {
    int memSize = sizeof(double) * getInputNeurons();
    std::memcpy(inputs, input, memSize);
    checkCudaErrors(cudaMemcpy(dInputs, input, memSize, cudaMemcpyHostToDevice));
}

double *GpuNetwork::getInput() {
    return inputs;
}

double *GpuNetwork::getOutput() {
    return inputs + noNeurons - getOutputNeurons();
}

int GpuNetwork::getAllNeurons() {
    return noNeurons;
}

int GpuNetwork::getInputOffset(int layer) {
    return neuronsUpToLayerCache[layer];
}

double* GpuNetwork::getWeights() {
    return weights;
}

int GpuNetwork::getWeightsOffset(int layer) {
    return weightsUpToLayerCache[layer];
}

double* GpuNetwork::getBiasValues() {
    return bias;
}
