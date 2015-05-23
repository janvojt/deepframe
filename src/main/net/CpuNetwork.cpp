/* 
 * File:   CpuNetwork.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "CpuNetwork.h"

#include <cstring>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "../util/cpuDebugHelpers.h"
#include "../common.h"
#include "Layer.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
CpuNetwork<dType>::CpuNetwork(NetworkConfiguration<dType> *conf) : Network<dType>(conf) {
//    initWeights();
//    initInputs();
//    initBias();
//    reinit();
}

template <typename dType>
CpuNetwork<dType>::CpuNetwork(const CpuNetwork& orig) : Network<dType>(orig.conf) {
    initWeights();
    initInputs();
    initBias();
    
    std::memcpy(this->inputs, orig.inputs, sizeof(dType) * this->noNeurons);
    std::memcpy(this->weights, orig.weights, sizeof(dType) * this->weightsUpToLayerCache[this->noLayers]);
    std::memcpy(this->bias, orig.bias, sizeof(dType) * this->getInputNeurons());
}

template <typename dType>
CpuNetwork<dType>::~CpuNetwork() {
    delete[] this->weightsUpToLayerCache;
    delete[] this->neuronsUpToLayerCache;
    delete[] this->weights;
    delete[] this->inputs;
    delete[] this->bias;
}

template <typename dType>
CpuNetwork<dType>* CpuNetwork<dType>::clone() {
    return new CpuNetwork(*this);
}

template <typename dType>
void CpuNetwork<dType>::merge(Network<dType>** nets, int size) {
    
    int noWeights = this->weightsUpToLayerCache[this->noLayers];
    for (int i = 0; i<size; i++) {
        
        // add weights
        dType *oWeights = nets[i]->getWeights();
        for (int j = 0; j<noWeights; j++) {
            this->weights[j] += oWeights[j];
        }
        
        // add bias
        dType *oBias = nets[i]->getBiasValues();
        for (int j = 0; j<this->noNeurons; j++) {
            this->bias[j] += oBias[j];
        }
    }
    
    // divide to get the average
    for (int j = 0; j<noWeights; j++) {
        this->weights[j] /= size+1;
    }
    for (int j = 0; j<this->noNeurons; j++) {
        this->bias[j] /= size+1;
    }
}

template <typename dType>
void CpuNetwork<dType>::reinit() {
    
    LOG()->info("Randomly initializing weights within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
    
    // overwrite weights with random doubles
    randomizeDoubles(&this->weights, this->weightsUpToLayerCache[this->noLayers]);
    
    if (this->conf->getBias()) {
    
        LOG()->info("Randomly initializing bias within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
        if (this->bias == NULL) {
            initBias();
        }
        
        // overwrite bias with random doubles
        randomizeDoubles(&this->bias, this->noNeurons);
    }
}


template <typename dType>
void CpuNetwork<dType>::initWeights() {
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
}

template <typename dType>
void CpuNetwork<dType>::initInputs() {
    int noNeurons = 0;
    this->neuronsUpToLayerCache = new int[this->noLayers+1];
    this->neuronsUpToLayerCache[0] = noNeurons;
    for (int i = 0; i<this->noLayers; i++) {
        noNeurons += this->conf->getNeurons(i);
        this->neuronsUpToLayerCache[i+1] = noNeurons;
    }
    this->noNeurons = noNeurons;
    this->inputs = new dType[noNeurons];
}

template <typename dType>
void CpuNetwork<dType>::initBias() {
    if (this->conf->getBias()) {
        this->bias = new dType[this->noNeurons];
    } else {
        this->bias = NULL;
    }
}

template<typename dType>
void CpuNetwork<dType>::allocateMemory() {
    std::cout << "Allocating memory for inputs: " << this->inputsCount << std::endl;
    this->inputs = new dType[this->inputsCount];
    this->weights = new dType[this->weightsCount];
}

template <typename dType>
void CpuNetwork<dType>::run() {
    
    for (int i = 1; i < this->noLayers; i++) {
        LOG()->debug("Computing forward run for layer %d.", i);
        this->layers[i]->forward();
    }
}

template <typename dType>
void CpuNetwork<dType>::randomizeDoubles(dType** memPtr, int size) {
    *memPtr = new dType[size];
    dType *mem = *memPtr;
    dType min = this->conf->getInitMin();
    dType max = this->conf->getInitMax();
    dType interval = max - min;
    for (int i = 0; i < size; i++) {
        mem[i] = ((dType) (rand()) / RAND_MAX * interval) + min;
    }
}

template <typename dType>
void CpuNetwork<dType>::applyBias(int l) {
    int n = this->conf->getNeurons(l);
    int offset = getInputOffset(l);
    for (int i = 0; i<n; i++) {
        this->inputs[offset + i] += this->bias[offset + i];
    }
}

template <typename dType>
void CpuNetwork<dType>::clearLayer(dType *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}

template <typename dType>
void CpuNetwork<dType>::setInput(dType* input) {
    std::memcpy(this->inputs, input, this->layers[0]->getOutputCount());
}

template <typename dType>
dType *CpuNetwork<dType>::getInputs() {
    return this->inputs;
}

template <typename dType>
dType *CpuNetwork<dType>::getInput() {
    return this->inputs;
}

template <typename dType>
dType *CpuNetwork<dType>::getOutput() {
    return this->layers[this->noLayers-1]->getInputs();
}

template <typename dType>
int CpuNetwork<dType>::getAllNeurons() {
    return this->noNeurons;
}

template <typename dType>
int CpuNetwork<dType>::getInputOffset(int layer) {
    return this->neuronsUpToLayerCache[layer];
}

template <typename dType>
dType* CpuNetwork<dType>::getWeights() {
    return this->weights;
}

template <typename dType>
int CpuNetwork<dType>::getWeightsOffset(int layer) {
    return this->weightsUpToLayerCache[layer];
}

template <typename dType>
dType* CpuNetwork<dType>::getBiasValues() {
    return this->bias;
}

INSTANTIATE_DATA_CLASS(CpuNetwork);