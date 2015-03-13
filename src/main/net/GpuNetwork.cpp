/* 
 * File:   GpuNetwork.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "GpuNetwork.h"

#include <cstring>
#include <string>
#include <stdlib.h>

#include "GpuConfiguration.h"

#include "../util/cudaDebugHelpers.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
GpuNetwork<dType>::GpuNetwork(NetworkConfiguration<dType> *netConf, GpuConfiguration *gpuConf) : Network<dType>(netConf) {
    cublasCreate(&this->cublasHandle);
    this->gpuConf = gpuConf;
    initWeights();
    initInputs();
    initBias();
    reinit();
}

template <typename dType>
GpuNetwork<dType>::GpuNetwork(const GpuNetwork& orig) : Network<dType>(orig.conf) {
    
    // initialize network and allocate memory
    cublasCreate(&this->cublasHandle);
    this->gpuConf = orig.gpuConf;
    initWeights();
    initInputs();
    initBias();
    
    // copy data
    int wMemSize = sizeof(dType) * getWeightsOffset(this->noLayers);
    int iMemSize = sizeof(dType) * getInputOffset(this->noLayers);
    checkCudaErrors(cudaMemcpy(this->dInputs, orig.dInputs, iMemSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->weights, orig.weights, wMemSize, cudaMemcpyDeviceToDevice));
    if (this->conf->getBias()) checkCudaErrors(cudaMemcpy(this->bias, orig.bias, iMemSize, cudaMemcpyDeviceToDevice));
}

template <typename dType>
GpuNetwork<dType>::~GpuNetwork() {
    cublasDestroy(this->cublasHandle);
    cudaFree(this->weights);
    cudaFree(this->dInputs);
    cudaFree(this->bias);
    delete[] this->weightsUpToLayerCache;
    delete[] this->neuronsUpToLayerCache;
    delete[] this->input;
    delete[] this->output;
}

template <typename dType>
GpuNetwork<dType>* GpuNetwork<dType>::clone() {
    return new GpuNetwork<dType>(*this);
}

template <typename dType>
void GpuNetwork<dType>::merge(Network<dType>** nets, int size) {
    
    int noWeights = this->weightsUpToLayerCache[this->noLayers];
    for (int i = 0; i<size; i++) {
        
        // add weights
        k_sumVectors(this->weights, nets[i]->getWeights(), noWeights);
        
        // add bias
        k_sumVectors(this->bias, nets[i]->getBiasValues(), this->noNeurons);
    }
    
    // divide to get the average
    k_divideVector(this->weights, size+1, noWeights);
    k_divideVector(this->bias, size+1, this->noNeurons);
}

template <typename dType>
void GpuNetwork<dType>::reinit() {
    
    LOG()->info("Randomly initializing weights and bias within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
    
    // overwrite weights with random doubles
    randomizeDoublesOnGpu(&this->weights, this->weightsUpToLayerCache[this->noLayers]);
    
    // overwrite bias with random doubles
    if (this->conf->getBias()) {
    
        LOG()->info("Randomly initializing bias within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
        if (this->bias == NULL) {
            initBias();
        }
        randomizeDoublesOnGpu(&this->bias, this->noNeurons);
    }
}

template <typename dType>
void GpuNetwork<dType>::randomizeDoublesOnGpu(dType **dMemory, int size) {
    
    // Initialize random values on GPU device memory.
    k_generateUniform(*this->gpuConf->getRandGen(), *dMemory, size);
    k_spreadInterval(this->conf->getInitMin(), this->conf->getInitMax(), *dMemory, size);
}

template <typename dType>
void GpuNetwork<dType>::initWeights() {
    int noWeights = 0;
    int pLayer = 1; // neurons in previous layer
    this->weightsUpToLayerCache = new int[this->noLayers+1];
    this->weightsUpToLayerCache[0] = noWeights;
    for (int i = 0; i<this->noLayers; i++) {
        int tLayer = this->conf->getNeurons(i);
        noWeights += pLayer * tLayer;
        this->weightsUpToLayerCache[i+1] = noWeights;
        pLayer = tLayer;
    }
    
    // use GPU random init,
    int memSize = sizeof(dType) * noWeights;
    checkCudaErrors(cudaMalloc(&this->weights, memSize));
}

template <typename dType>
void GpuNetwork<dType>::initInputs() {
    int noNeurons = 0;
    this->neuronsUpToLayerCache = new int[this->noLayers+1];
    this->neuronsUpToLayerCache[0] = noNeurons;
    for (int i = 0; i<this->noLayers; i++) {
        noNeurons += this->conf->getNeurons(i);
        this->neuronsUpToLayerCache[i+1] = noNeurons;
    }
    this->noNeurons = noNeurons;
    
    // allocate host memory
    this->input = new dType[this->getInputNeurons()];
    this->output = new dType[this->getOutputNeurons()];
    
    // allocate device memory
    int memSize = sizeof(dType) * noNeurons;
    checkCudaErrors(cudaMalloc(&dInputs, memSize));
//    // initialize allocated device memory
//    checkCudaErrors(cudaMemset(dInputs, 0, memSize));
}

template <typename dType>
void GpuNetwork<dType>::initBias() {
    if (this->conf->getBias()) {
        int memSize = sizeof(dType) * this->noNeurons;
        checkCudaErrors(cudaMalloc(&this->bias, memSize));
    } else {
        this->bias = NULL;
    }
}

template <typename dType>
void GpuNetwork<dType>::run() {
    // number of neurons in so far processed layers
    dType *dWeightsPtr = this->weights + this->getInputNeurons();
    dType *dInputsPtr = this->dInputs;
    dType *dBiasPtr = this->bias;
    
    // copy weights and bias from host to device
//    int wMemSize = sizeof(dType) * getWeightsOffset(noLayers);
//    int iMemSize = sizeof(dType) * getInputOffset(noLayers);
//    checkCudaErrors(cudaMemcpy(dInputs, inputs, iMemSize, cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(dWeights, weights, wMemSize, cudaMemcpyHostToDevice));
//    if (conf->getBias()) checkCudaErrors(cudaMemcpy(dBias, bias, iMemSize, cudaMemcpyHostToDevice));
    
    // for every layer
    for (int l = 0; l<this->noLayers-1; l++) {
        int nThisLayer = this->conf->getNeurons(l);
        int nNextLayer = this->conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        int nextLayerSize = nNextLayer * sizeof(dType);
        cudaMemset(dInputsPtr + nThisLayer, 0.0, nextLayerSize);
        
        //note cuBLAS is column primary!
        //need to transpose the order
        const dType alpha = 1.0;
        const dType beta = 0.0;
        k_gemm(this->cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                1, nNextLayer, nThisLayer,
                &alpha, dInputsPtr, 1,
                dWeightsPtr, nNextLayer,
                &beta, dInputsPtr+nThisLayer, 1);

        if (this->conf->getBias()) {
            k_sumVectors(dInputsPtr + nThisLayer, dBiasPtr + nThisLayer, nNextLayer);
            dBiasPtr += nThisLayer;
        }
        
        k_computeSigmoid(dInputsPtr + nThisLayer, nNextLayer);
	
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
}

template <typename dType>
void GpuNetwork<dType>::setInput(dType* input) {
//    compare('b', dInputs, inputs, getInputOffset(noLayers));
    int memSize = sizeof(dType) * this->getInputNeurons();
    std::memcpy(this->input, input, memSize);
    checkCudaErrors(cudaMemcpy(this->dInputs, input, memSize, cudaMemcpyHostToDevice));
//    compare('a', dInputs, inputs, getInputOffset(noLayers));
}

template <typename dType>
dType *GpuNetwork<dType>::getInputs() {
    return this->dInputs;
}

template <typename dType>
dType *GpuNetwork<dType>::getInput() {
    return this->input;
}

template <typename dType>
dType *GpuNetwork<dType>::getOutput() {
    
    // copy network output to host
    dType *dOutput = this->dInputs + getInputOffset(this->noLayers-1);
    int oMemSize = this->getOutputNeurons() * sizeof(dType);
    checkCudaErrors(cudaMemcpy(this->output, dOutput, oMemSize, cudaMemcpyDeviceToHost));
    
    return this->output;
}

template <typename dType>
int GpuNetwork<dType>::getAllNeurons() {
    return this->noNeurons;
}

template <typename dType>
int GpuNetwork<dType>::getInputOffset(int layer) {
    return this->neuronsUpToLayerCache[layer];
}

template <typename dType>
dType* GpuNetwork<dType>::getWeights() {
    return this->weights;
}

template <typename dType>
int GpuNetwork<dType>::getWeightsOffset(int layer) {
    return this->weightsUpToLayerCache[layer];
}

template <typename dType>
dType* GpuNetwork<dType>::getBiasValues() {
    return this->bias;
}

INSTANTIATE_DATA_CLASS(GpuNetwork);
