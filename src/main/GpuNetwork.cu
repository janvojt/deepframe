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


GpuNetwork::GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf) : Network(netConf) {
    this->gpuConf = gpuConf;
    initWeights();
    initInputs();
    initBias();
}

GpuNetwork::GpuNetwork(const GpuNetwork& orig) : Network(orig) {
}

GpuNetwork::~GpuNetwork() {
    cudaFree(dWeights);
    cudaFree(dBias);
    delete[] weightsUpToLayerCache;
    delete[] neuronsUpToLayerCache;
    delete[] weights;
    delete[] inputs;
    delete[] bias;
}

void GpuNetwork::randomizeDoublesOnGpu(double **hMemory, double **dMemory, int size) {

    cudaError_t error;
    int memSize = sizeof(double) * size;
    error = cudaMalloc(dMemory, memSize);

    if (error != cudaSuccess) {
        LOG()->error("Error when trying to cudaMalloc memory, error code: %d.", error);
        exit(EXIT_FAILURE);
    }
    
    // Initialize random values on GPU device memory.
    curandGenerateUniformDouble(*gpuConf->getRandGen(), *dMemory, size);

    // Copy to host memory.
    *hMemory = new double[size];
    cudaMemcpy(static_cast<void*>(*hMemory), static_cast<void*>(*dMemory), memSize, cudaMemcpyDeviceToHost);
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
    
    randomizeDoublesOnGpu(&weights, &dWeights, noWeights);
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
    inputs = new double[noNeurons];
}

void GpuNetwork::initBias() {
    if (conf->getBias()) {
        bias = new double[noNeurons];
        
        // Initialize bias.
        randomizeDoublesOnGpu(&bias, &dBias, noNeurons);
        
    } else {
        bias = NULL;
    }
}

void GpuNetwork::run() {
    // number of neurons in so far processed layers
    int nPrevLayers = 0;
    double *weighPtr = weights + getInputNeurons();
    
    // for every layer
    for (int l = 0; l<noLayers-1; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        clearLayer(inputs + nPrevLayers + nThisLayer, nNextLayer);
        
        // for every neuron in (l)th layer
        for (int i = 0; i<nThisLayer; i++) {
            int indexFrom = nPrevLayers + i;
            // for every neuron in (l+1)th layer
            for (int j = 0; j<nNextLayer; j++) {
                int indexTo = nPrevLayers + nThisLayer + j;
                inputs[indexTo] += *weighPtr * inputs[indexFrom];
                weighPtr++;
            }
        }

        if (conf->getBias()) {
            applyBias(l+1);
        }
        
        // Run through activation function
        conf->activationFnc(inputs+nPrevLayers+nThisLayer, inputs+nPrevLayers+nThisLayer, nNextLayer);
        
        nPrevLayers += nThisLayer;
    }
}

void GpuNetwork::applyBias(int l) {
    int n = conf->getNeurons(l);
    int offset = getInputOffset(l);
    for (int i = 0; i<n; i++) {
        inputs[offset + i] += bias[offset + i];
    }
}

void GpuNetwork::clearLayer(double *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}

void GpuNetwork::setInput(double* input) {
    std::memcpy(inputs, input, sizeof(double) * getInputNeurons());
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
