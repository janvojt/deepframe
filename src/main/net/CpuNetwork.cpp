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
    std::memcpy(this->outputDiffs, orig.outputDiffs, sizeof(data_t) * this->inputsCount);
    std::memcpy(this->weights, orig.weights, sizeof(data_t) * this->weightsCount);
    std::memcpy(this->weightDiffs, orig.weightDiffs, sizeof(data_t) * this->weightsCount);
}

CpuNetwork::~CpuNetwork() {
    delete[] this->weights;
    delete[] this->weightDiffs;
    delete[] this->inputs;
    delete[] this->outputDiffs;
}

CpuNetwork* CpuNetwork::clone() {
    return new CpuNetwork(*this);
}

void CpuNetwork::merge(Network** nets, int size) {
    
    LOG()->warn("Method for merging CPU networks is not maintained and likely not working correctly.");
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
    this->outputDiffs = new data_t[this->inputsCount];
    
    LOG()->debug("Allocating memory for %d weights.", this->weightsCount);
    this->weights = new data_t[this->weightsCount];
    this->weightDiffs = new data_t[this->weightsCount];
}


void CpuNetwork::forward() {
    for (int i = 1; i < this->noLayers; i++) {
//        LOG()->debug("Computing forward run on CPU for layer %d.", i);
        this->layers[i]->forwardCpu();
    }
}

void CpuNetwork::backward() {
    
    LOG()->debug("Computing output gradients for last layer on CPU.");
    this->layers[noLayers-1]->backwardLastCpu(expectedOutput);
    
    for (int i = noLayers-1; i > 0; i--) {
        LOG()->debug("Computing backward run on CPU for layer %d.", i);
        this->layers[i]->backwardCpu();
    }
    
    // update all weights
    for (int i = 0; i<weightsCount; i++) {
        weights[i] += weightDiffs[i];
    }
}

void CpuNetwork::setInput(data_t* input) {
    processInput(input);
    std::memcpy(this->inputs, input, getInputNeurons() * sizeof(data_t));
}

data_t *CpuNetwork::getInput() {
    return this->inputs;
}

data_t *CpuNetwork::getOutput() {
    return this->layers[this->noLayers-1]->getOutputs();
}

void CpuNetwork::setExpectedOutput(data_t* output) {
    expectedOutput = output;
}