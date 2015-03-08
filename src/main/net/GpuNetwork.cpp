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

#include "../util/cudaHelpers.h"
#include "../util/cudaDebugHelpers.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"


GpuNetwork::GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf) : Network(netConf) {
    cublasCreate(&cublasHandle);
    this->gpuConf = gpuConf;
    initWeights();
    initInputs();
    initBias();
    reinit();
}

GpuNetwork::GpuNetwork(const GpuNetwork& orig) : Network(orig.conf) {
    
    // initialize network and allocate memory
    cublasCreate(&cublasHandle);
    this->gpuConf = orig.gpuConf;
    initWeights();
    initInputs();
    initBias();
    
    // copy data
    int wMemSize = sizeof(double) * getWeightsOffset(noLayers);
    int iMemSize = sizeof(double) * getInputOffset(noLayers);
    checkCudaErrors(cudaMemcpy(dInputs, orig.dInputs, iMemSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(weights, orig.weights, wMemSize, cudaMemcpyDeviceToDevice));
    if (conf->getBias()) checkCudaErrors(cudaMemcpy(bias, orig.bias, iMemSize, cudaMemcpyDeviceToDevice));
}

GpuNetwork::~GpuNetwork() {
    cublasDestroy(cublasHandle);
    cudaFree(weights);
    cudaFree(dInputs);
    cudaFree(bias);
    delete[] weightsUpToLayerCache;
    delete[] neuronsUpToLayerCache;
    delete[] input;
    delete[] output;
}

GpuNetwork* GpuNetwork::clone() {
    return new GpuNetwork(*this);
}

void GpuNetwork::merge(Network** nets, int size) {
    
    int noWeights = weightsUpToLayerCache[noLayers];
    for (int i = 0; i<size; i++) {
        
        // add weights
        k_sumVectors(weights, nets[i]->getWeights(), noWeights);
        
        // add bias
        k_sumVectors(bias, nets[i]->getBiasValues(), noNeurons);
    }
    
    // divide to get the average
    k_divideVector(weights, size+1, noWeights);
    k_divideVector(bias, size+1, noNeurons);
}

void GpuNetwork::reinit() {
    
    LOG()->info("Randomly initializing weights and bias within the interval (%f,%f).", conf->getInitMin(), conf->getInitMax());
    
    // overwrite weights with random doubles
    randomizeDoublesOnGpu(&weights, weightsUpToLayerCache[noLayers]);
    
    // overwrite bias with random doubles
    if (conf->getBias()) {
    
        LOG()->info("Randomly initializing bias within the interval (%f,%f).", conf->getInitMin(), conf->getInitMax());
        if (bias == NULL) {
            initBias();
        }
        randomizeDoublesOnGpu(&bias, noNeurons);
    }
}

void GpuNetwork::randomizeDoublesOnGpu(double **dMemory, int size) {
    
    // Initialize random values on GPU device memory.
    curandGenerateUniformDouble(*gpuConf->getRandGen(), *dMemory, size);
    k_spreadInterval(conf->getInitMin(), conf->getInitMax(), *dMemory, size);
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
    
    // use GPU random init,
    int memSize = sizeof(double) * noWeights;
    checkCudaErrors(cudaMalloc(&weights, memSize));
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
    input = new double[this->getInputNeurons()];
    output = new double[this->getOutputNeurons()];
    
    // allocate device memory
    int memSize = sizeof(double) * noNeurons;
    checkCudaErrors(cudaMalloc(&dInputs, memSize));
//    // initialize allocated device memory
//    checkCudaErrors(cudaMemset(dInputs, 0, memSize));
}

void GpuNetwork::initBias() {
    if (conf->getBias()) {
        int memSize = sizeof(double) * noNeurons;
        checkCudaErrors(cudaMalloc(&bias, memSize));
    } else {
        bias = NULL;
    }
}

void GpuNetwork::run() {
    // number of neurons in so far processed layers
    double *dWeightsPtr = weights + getInputNeurons();
    double *dInputsPtr = dInputs;
    double *dBiasPtr = bias;
    
    // copy weights and bias from host to device
//    int wMemSize = sizeof(double) * getWeightsOffset(noLayers);
//    int iMemSize = sizeof(double) * getInputOffset(noLayers);
//    checkCudaErrors(cudaMemcpy(dInputs, inputs, iMemSize, cudaMemcpyHostToDevice));
//    checkCudaErrors(cudaMemcpy(dWeights, weights, wMemSize, cudaMemcpyHostToDevice));
//    if (conf->getBias()) checkCudaErrors(cudaMemcpy(dBias, bias, iMemSize, cudaMemcpyHostToDevice));
    
    // for every layer
    for (int l = 0; l<noLayers-1; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        int nextLayerSize = nNextLayer * sizeof(double);
        cudaMemset(dInputsPtr + nThisLayer, 0.0, nextLayerSize);
        
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

void GpuNetwork::setInput(double* input) {
//    compare('b', dInputs, inputs, getInputOffset(noLayers));
    int memSize = sizeof(double) * getInputNeurons();
    std::memcpy(this->input, input, memSize);
    checkCudaErrors(cudaMemcpy(dInputs, input, memSize, cudaMemcpyHostToDevice));
//    compare('a', dInputs, inputs, getInputOffset(noLayers));
}

double *GpuNetwork::getInputs() {
    return dInputs;
}

double *GpuNetwork::getInput() {
    return input;
}

double *GpuNetwork::getOutput() {
    
    // copy network output to host
    double *dOutput = dInputs + getInputOffset(noLayers-1);
    int oMemSize = getOutputNeurons() * sizeof(double);
    checkCudaErrors(cudaMemcpy(this->output, dOutput, oMemSize, cudaMemcpyDeviceToHost));
    
    return this->output;
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
