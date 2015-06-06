/* 
 * File:   FullyConnectedLayer.cpp
 * Author: janvojt
 * 
 * Created on May 17, 2015, 12:55 AM
 */

#include "FullyConnectedLayer.h"

#include <algorithm>
#include <sstream>
#include "../../common.h"
#include "../LayerFactory.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FullyConnectedLayer::FullyConnectedLayer() {
}

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedLayer& orig) {
}

FullyConnectedLayer::~FullyConnectedLayer() {
}

void FullyConnectedLayer::setup(Layer *previousLayer, string confString) {
    
    processConfString(confString);
    
    if (previousLayer != NULL) {
        // this is not the input layer
        this->previousLayer = previousLayer;
        this->weightsCount = previousLayer->getOutputsCount() * conf.outputSize;
        if (conf.useBias) {
            this->weightsCount += conf.outputSize;
        }
        previousLayer->setNextLayer(this);
    } else {
        this->weightsCount = 0;
    }
    this->inputsCount = conf.outputSize;
}

void FullyConnectedLayer::forwardCpu() {
    
    int inputSize = this->previousLayer->getOutputsCount();
    data_t *inputPtr = this->previousLayer->getInputs();
    data_t *outputPtr = this->getInputs();
    
    // Clear output neurons
    std::fill_n(outputPtr, conf.outputSize, 0);
    
    data_t *weightPtr = this->weights;
    for (int i = 0; i<inputSize; i++) {
        for (int j = 0; j<conf.outputSize; j++) {
            outputPtr[j] += inputPtr[i] * *weightPtr;
            weightPtr++;
        }
    }
    
    // Apply bias
    if (conf.useBias) {
        for (int i = 0; i++; i<conf.outputSize) {
            outputPtr[i] += *weightPtr;
            weightPtr++;
        }
    }
    
    // Run through activation function
    conf.activationFnc(outputPtr, outputPtr, conf.outputSize);
}

void FullyConnectedLayer::forwardGpu() {
    //TODO
}


void FullyConnectedLayer::backwardCpu() {
    //TODO
}


void FullyConnectedLayer::backwardGpu() {
    //TODO
}

void FullyConnectedLayer::processConfString(string confString) {
    // dummy variable for delimiters
    char sep;
    istringstream iss (confString);
    
    if (!(iss >> conf.outputSize)) {
        LOG()->error("Could not read output size for FullyConnected layer.");
    }
    
    iss >> sep;
    
    if (!(iss >> boolalpha >> conf.useBias)) {
        LOG()->warn("Could not read bias for FullyConnected layer from configuration. Not using bias...");
        conf.useBias = false;
    }
    cout << conf.useBias;
}

static LayerRegister<FullyConnectedLayer> reg("FullyConnected");