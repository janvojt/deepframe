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

GpuNetwork::GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf) : Network(netConf) {
    cublasCreate(&this->cublasHandle);
    this->gpuConf = gpuConf;
}

GpuNetwork::GpuNetwork(const GpuNetwork& orig) : Network(orig.conf) {
    
    // initialize network and allocate memory
    cublasCreate(&this->cublasHandle);
    this->gpuConf = orig.gpuConf;
    
    this->allocateMemory();
    
    // copy data
    int wMemSize = sizeof(data_t) * this->weightsCount;
    int iMemSize = sizeof(data_t) * this->inputsCount;
    checkCudaErrors(cudaMemcpy(this->inputs, orig.inputs, iMemSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->weights, orig.weights, wMemSize, cudaMemcpyDeviceToDevice));
}

GpuNetwork::~GpuNetwork() {
    cublasDestroy(this->cublasHandle);
    cudaFree(this->weights);
    delete[] this->input;
    delete[] this->output;
}

GpuNetwork* GpuNetwork::clone() {
    return new GpuNetwork(*this);
}

void GpuNetwork::merge(Network** nets, int size) {
    
    int noWeights = this->weightsCount;
    for (int i = 0; i<size; i++) {
        
        // add weights
        k_sumVectors(this->weights, nets[i]->getWeights(), noWeights);
    }
    
    // divide to get the average
    k_divideVector(this->weights, size+1, noWeights);
}

void GpuNetwork::reinit() {
    LOG()->info("Randomly initializing weights within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
    k_generateUniform(*this->gpuConf->getRandGen(), this->weights, this->weightsCount);
    k_spreadInterval(this->conf->getInitMin(), this->conf->getInitMax(), this->weights, this->weightsCount);
}

void GpuNetwork::allocateMemory() {
    checkCudaErrors(cudaMalloc(&this->weights, sizeof(data_t) * this->weightsCount));
    checkCudaErrors(cudaMalloc(&this->inputs, sizeof(data_t) * this->inputsCount));
}

//void GpuNetwork::forward() {
//    // number of neurons in so far processed layers
//    data_t *dWeightsPtr = this->weights + this->getInputNeurons();
//    data_t *dInputsPtr = this->inputs;
//    
//    // copy weights and bias from host to device
////    int wMemSize = sizeof(data_t) * getWeightsOffset(noLayers);
////    int iMemSize = sizeof(data_t) * getInputOffset(noLayers);
////    checkCudaErrors(cudaMemcpy(dInputs, inputs, iMemSize, cudaMemcpyHostToDevice));
////    checkCudaErrors(cudaMemcpy(dWeights, weights, wMemSize, cudaMemcpyHostToDevice));
////    if (conf->getBias()) checkCudaErrors(cudaMemcpy(dBias, bias, iMemSize, cudaMemcpyHostToDevice));
//    
//    // for every layer
//    for (int l = 0; l<this->noLayers-1; l++) {
//        int nThisLayer = this->conf->getNeurons(l);
//        int nNextLayer = this->conf->getNeurons(l+1);
//        
//        // clear the following layer just before working with it
//        int nextLayerSize = nNextLayer * sizeof(data_t);
//        cudaMemset(dInputsPtr + nThisLayer, 0.0, nextLayerSize);
//        
//        //note cuBLAS is column primary!
//        //need to transpose the order
//        const data_t alpha = 1.0;
//        const data_t beta = 0.0;
//        k_gemm(this->cublasHandle,
//                CUBLAS_OP_N, CUBLAS_OP_T,
//                1, nNextLayer, nThisLayer,
//                &alpha, dInputsPtr, 1,
//                dWeightsPtr, nNextLayer,
//                &beta, dInputsPtr+nThisLayer, 1);
//        
//        k_computeSigmoid(dInputsPtr + nThisLayer, nNextLayer);
//	
//        dWeightsPtr += nThisLayer * nNextLayer;
//        dInputsPtr += nThisLayer;
//    }
//    
//    // copy all weights and bias back to host
////    checkCudaErrors(cudaMemcpy(inputs, dInputs, iMemSize, cudaMemcpyDeviceToHost));
////    checkCudaErrors(cudaMemcpy(weights, dWeights, wMemSize, cudaMemcpyDeviceToHost));
////    if (conf->getBias()) checkCudaErrors(cudaMemcpy(bias, dBias, iMemSize, cudaMemcpyDeviceToHost));
//    
////    compare('w', dWeights, weights, getWeightsOffset(noLayers));
////    if (conf->getBias()) compare('b', dBias, bias, getInputOffset(noLayers));
////    compare('i', dInputs, inputs, getInputOffset(noLayers));
//}

void GpuNetwork::setInput(data_t* input) {
//    compare('b', dInputs, inputs, getInputOffset(noLayers));
    int memSize = sizeof(data_t) * this->getInputNeurons();
    std::memcpy(this->input, input, memSize);
    checkCudaErrors(cudaMemcpy(this->inputs, input, memSize, cudaMemcpyHostToDevice));
//    compare('a', dInputs, inputs, getInputOffset(noLayers));
}

data_t *GpuNetwork::getInput() {
    return this->input;
}

data_t *GpuNetwork::getOutput() {
    
    // copy network output to host
    data_t *dOutput = this->inputs + this->layers[this->noLayers-1]->getOutputsCount();
    int oMemSize = this->getOutputNeurons() * sizeof(data_t);
    checkCudaErrors(cudaMemcpy(this->output, dOutput, oMemSize, cudaMemcpyDeviceToHost));
    
    return this->output;
}

bool GpuNetwork::useGpu() {
    return true;
}
