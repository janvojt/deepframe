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
}

template <typename dType>
CpuNetwork<dType>::CpuNetwork(const CpuNetwork& orig) : Network<dType>(orig.conf) {
    this->allocateMemory();
    std::memcpy(this->inputs, orig.inputs, sizeof(dType) * this->inputsCount);
    std::memcpy(this->weights, orig.weights, sizeof(dType) * this->weightsCount);
}

template <typename dType>
CpuNetwork<dType>::~CpuNetwork() {
    delete[] this->weights;
    delete[] this->inputs;
}

template <typename dType>
CpuNetwork<dType>* CpuNetwork<dType>::clone() {
    return new CpuNetwork(*this);
}

template <typename dType>
void CpuNetwork<dType>::merge(Network<dType>** nets, int size) {
    
    int noWeights = this->weightsCount;
    for (int i = 0; i<size; i++) {
        
        // add weights
        dType *oWeights = nets[i]->getWeights();
        for (int j = 0; j<noWeights; j++) {
            this->weights[j] += oWeights[j];
        }
    }
    
    // divide to get the average
    for (int j = 0; j<noWeights; j++) {
        this->weights[j] /= size+1;
    }
}

template <typename dType>
void CpuNetwork<dType>::reinit() {
    LOG()->info("Randomly initializing weights within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
    dType min = this->conf->getInitMin();
    dType max = this->conf->getInitMax();
    dType interval = max - min;
    for (int i = 0; i < this->weightsCount; i++) {
        this->weights[i] = ((dType) (rand()) / RAND_MAX * interval) + min;
    }
}

template<typename dType>
void CpuNetwork<dType>::allocateMemory() {
    LOG()->debug("Allocating memory for %d inputs.", this->inputsCount);
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
void CpuNetwork<dType>::setInput(dType* input) {
    std::memcpy(this->inputs, input, this->layers[0]->getOutputCount());
}

template <typename dType>
dType *CpuNetwork<dType>::getInput() {
    return this->inputs;
}

template <typename dType>
dType *CpuNetwork<dType>::getOutput() {
    return this->layers[this->noLayers-1]->getInputs();
}

INSTANTIATE_DATA_CLASS(CpuNetwork);