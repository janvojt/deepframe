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

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
CpuNetwork<dType>::CpuNetwork(NetworkConfiguration<dType> *conf) : Network<dType>(conf) {
    initWeights();
    initInputs();
    initBias();
    reinit();
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
    inputs = new dType[noNeurons];
}

template <typename dType>
void CpuNetwork<dType>::initBias() {
    if (this->conf->getBias()) {
        this->bias = new dType[this->noNeurons];
    } else {
        this->bias = NULL;
    }
}

template <typename dType>
void CpuNetwork<dType>::run() {
    // number of neurons in so far processed layers
    int nPrevLayers = 0;
    dType *weighPtr = this->weights + this->getInputNeurons();
    
    // for every layer
    for (int l = 0; l<this->noLayers-1; l++) {
        int nThisLayer = this->conf->getNeurons(l);
        int nNextLayer = this->conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        clearLayer(this->inputs + nPrevLayers + nThisLayer, nNextLayer);
        
        // for every neuron in (l)th layer
        for (int i = 0; i<nThisLayer; i++) {
            int indexFrom = nPrevLayers + i;
            // for every neuron in (l+1)th layer
            for (int j = 0; j<nNextLayer; j++) {
                int indexTo = nPrevLayers + nThisLayer + j;
                this->inputs[indexTo] += *weighPtr * this->inputs[indexFrom];
                weighPtr++;
            }
        }

        if (this->conf->getBias()) {
            applyBias(l+1);
        }
        
        // Run through activation function
        this->conf->activationFnc(this->inputs+nPrevLayers+nThisLayer, this->inputs+nPrevLayers+nThisLayer, nNextLayer);
        
        nPrevLayers += nThisLayer;
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
    std::memcpy(this->inputs, input, sizeof(dType) * this->getInputNeurons());
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
    return this->inputs + this->neuronsUpToLayerCache[this->noLayers-1];
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