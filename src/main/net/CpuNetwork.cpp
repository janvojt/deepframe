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

CpuNetwork::CpuNetwork(NetworkConfiguration *conf) : Network(conf) {
}

CpuNetwork::CpuNetwork(const CpuNetwork& orig) : Network(orig.conf) {
    this->allocateMemory();
    std::memcpy(this->inputs, orig.inputs, sizeof(data_t) * this->inputsCount);
    std::memcpy(this->weights, orig.weights, sizeof(data_t) * this->weightsCount);
}

CpuNetwork::~CpuNetwork() {
    delete[] this->weights;
    delete[] this->inputs;
}

CpuNetwork* CpuNetwork::clone() {
    return new CpuNetwork(*this);
}

void CpuNetwork::merge(Network** nets, int size) {
    
    int noWeights = this->weightsCount;
    for (int i = 0; i<size; i++) {
        
        // add weights
        data_t *oWeights = nets[i]->getWeights();
        for (int j = 0; j<noWeights; j++) {
            this->weights[j] += oWeights[j];
        }
    }
    
    // divide to get the average
    for (int j = 0; j<noWeights; j++) {
        this->weights[j] /= size+1;
    }
}

void CpuNetwork::reinit() {
    LOG()->info("Randomly initializing weights within the interval (%f,%f).", this->conf->getInitMin(), this->conf->getInitMax());
    data_t min = this->conf->getInitMin();
    data_t max = this->conf->getInitMax();
    data_t interval = max - min;
    for (int i = 0; i < this->weightsCount; i++) {
        this->weights[i] = ((data_t) (rand()) / RAND_MAX * interval) + min;
    }
}

void CpuNetwork::allocateMemory() {
    LOG()->debug("Allocating memory for %d inputs.", this->inputsCount);
    this->inputs = new data_t[this->inputsCount];
    this->weights = new data_t[this->weightsCount];
}

void CpuNetwork::run() {
    for (int i = 1; i < this->noLayers; i++) {
        LOG()->debug("Computing forward run for layer %d.", i);
        this->layers[i]->forward();
    }
}

void CpuNetwork::setInput(data_t* input) {
    std::memcpy(this->inputs, input, this->layers[0]->getOutputsCount());
}

data_t *CpuNetwork::getInput() {
    return this->inputs;
}

data_t *CpuNetwork::getOutput() {
    return this->layers[this->noLayers-1]->getInputs();
}
