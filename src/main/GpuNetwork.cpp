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

#include "util/cudaHelpers.h"
#include "util/cudaDebugHelpers.h"

#include <cstring>
#include <string>
#include <stdlib.h>

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
//    // initialize allocated device memory
//    checkCudaErrors(cudaMemset(dInputs, 0, memSize));
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
//    checkCudaErrors(cudaMemcpy(dInputs, inputs, iMemSize, cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(dWeights, weights, wMemSize, cudaMemcpyHostToDevice));
//    if (conf->getBias()) checkCudaErrors(cudaMemcpy(dBias, bias, iMemSize, cudaMemcpyHostToDevice));
    
    // for every layer
    for (int l = 0; l<noLayers-1; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        k_clearLayer(dim3(1), dim3(nNextLayer), dInputsPtr + nThisLayer);
        
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
            k_sumArrays(dim3(1), dim3(nNextLayer), dInputsPtr + nThisLayer, dBiasPtr + nThisLayer);
            dBiasPtr += nThisLayer;
        }
        
        k_computeSigmoid(dim3(1), dim3(nNextLayer), dInputsPtr + nThisLayer);
	
        dWeightsPtr += nThisLayer * nNextLayer;
        dInputsPtr += nThisLayer;
    }
    
    // copy all weights and bias back to host
//    checkCudaErrors(cudaMemcpy(inputs, dInputs, iMemSize, cudaMemcpyDeviceToHost));
//    checkCudaErrors(cudaMemcpy(weights, dWeights, wMemSize, cudaMemcpyDeviceToHost));
//    if (conf->getBias()) checkCudaErrors(cudaMemcpy(bias, dBias, iMemSize, cudaMemcpyDeviceToHost));
    
//    compare('w', dWeights, weights, getWeightsOffset(noLayers));
//    if (conf->getBias()) compare('b', dBias, bias, getInputOffset(noLayers));
//    compare('i', dInputs, inputs, getInputOffset(noLayers));
    
    // copy network output to host
    double *dOutput = dInputs + getInputOffset(noLayers-1);
    double *output = inputs + getInputOffset(noLayers-1);
    int oMemSize = getOutputNeurons() * sizeof(double);
    checkCudaErrors(cudaMemcpy(output, dOutput, oMemSize, cudaMemcpyDeviceToHost));
}

void GpuNetwork::setInput(double* input) {
//    compare('b', dInputs, inputs, getInputOffset(noLayers));
    int memSize = sizeof(double) * getInputNeurons();
    std::memcpy(inputs, input, memSize);
    checkCudaErrors(cudaMemcpy(dInputs, input, memSize, cudaMemcpyHostToDevice));
//    compare('a', dInputs, inputs, getInputOffset(noLayers));
}

double *GpuNetwork::getInputs() {
    return dInputs;
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
    return dWeights;
}

int GpuNetwork::getWeightsOffset(int layer) {
    return weightsUpToLayerCache[layer];
}

double* GpuNetwork::getBiasValues() {
    return dBias;
}
