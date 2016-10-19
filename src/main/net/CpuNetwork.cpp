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
#include <fstream>
#include <random>

#include "../util/cpuDebugHelpers.h"
#include "../common.h"
#include "Layer.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

using namespace std;

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

    if (conf->getImportFile() != NULL) {
        LOG()->info("Importing network parameters from file '%s'.", conf->getImportFile());
        ifstream fp(conf->getImportFile(), ios::in|ios::binary);
        if (fp.is_open()) {
            // parse dimension sizes
            data_t *w = weights;
            fp.read((char *) w, sizeof(data_t) * weightsCount);
            fp.close();
        } else {
            LOG()->error("Cannot open file '%s' for reading network parameters.", conf->getImportFile());
        }
        return;
    }

    LOG()->info("Randomly initializing weights within the interval (%f,%f).", conf->getInitMin(), conf->getInitMax());
    data_t min = conf->getInitMin();
    data_t max = conf->getInitMax();
    data_t interval = max - min;

    if (interval < 0) {
        // we are using Gaussian distribution with the given standard deviation
        std::default_random_engine gen;
        std::normal_distribution<data_t> dist(0., conf->getInitMin());
        for (int i = 0; i < this->weightsCount; i++) {
            this->weights[i] = dist(gen);
        }
    } else {
        // we are using uniform distribution with the given interval
        for (int i = 0; i < this->weightsCount; i++) {
            this->weights[i] = ((data_t) (rand()) / RAND_MAX * interval) + min;
        }
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

void CpuNetwork::save(char *filePath) {

    // open IDX file with the dataset
    ofstream fp(filePath, ios::out|ios::binary);

    if (fp.is_open()) {

        // write weights
        for (int i = 0; i<weightsCount; i++) {
            fp.write((char *) (weights+i), sizeof(data_t));
        }

        fp.close();

        LOG()->info("Serialized network parameters in file '%s'.", filePath);
    } else {
        LOG()->error("Cannot open file '%s' for writing.", filePath);
    }
}
