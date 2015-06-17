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

void FullyConnectedLayer::setup(string confString) {
    
    processConfString(confString);
    
    if (previousLayer != NULL) {
        // this is not the input layer
        this->weightsCount = previousLayer->getOutputsCount() * conf.outputSize;
        if (conf.useBias) {
            this->weightsCount += conf.outputSize;
        }
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
        for (int i = 0; i<conf.outputSize; i++) {
            outputPtr[i] += *weightPtr;
            weightPtr++;
        }
    }
    
    // Run through activation function
    netConf->activationFnc(outputPtr, outputPtr, conf.outputSize);
}

void FullyConnectedLayer::forwardGpu() {
    //TODO
}


void FullyConnectedLayer::backwardCpu() {
        
    // COMPUTE LOCAL GRADIENTS for this layer

    // compute derivatives of neuron inputs for this layer
    void (*daf) (data_t*,data_t*,int) = this->netConf->dActivationFnc;
    data_t *thisInputDerivatives = new data_t[inputsCount];
    daf(inputs, thisInputDerivatives, inputsCount);

    // compute local gradients for this layer
    int nextNeurons = nextLayer->getOutputsCount();
    data_t *nextOutputDiffs = nextLayer->getOutputDiffs();
    data_t *nextWeights = nextLayer->getWeights();
    for (int i = 0; i<inputsCount; i++) {
        data_t sumNextGradient = 0;
        for (int j = 0; j<nextNeurons; j++) {
            sumNextGradient += nextOutputDiffs[j] * nextWeights[i * nextNeurons + j];
        }
        outputDiffs[i] = sumNextGradient * thisInputDerivatives[i];
//            LOG()->debug("Local gradient for neuron [%d, %d] : %f.", l, i, thisLocalGradient[i]);
    }
    delete[] thisInputDerivatives;
//        dumpHostArray('l', thisLocalGradient, thisNeurons + nextNeurons);
    
    computeTotalDiffs();
}

void FullyConnectedLayer::backwardLastCpu(data_t* expectedOutput) {
    
    void (*daf) (data_t*,data_t*,int) = this->netConf->dActivationFnc;
    
    // compute local gradients
    data_t *dv = new data_t[inputsCount]; //TODO allocate only once
    daf(inputs, dv, inputsCount);
    for (int i = 0; i<inputsCount; i++) {
        outputDiffs[i] = (inputs[i] - expectedOutput[i]) * dv[i];
    }
    delete[] dv;
    
    computeTotalDiffs();
}

void FullyConnectedLayer::backwardGpu() {
    //TODO
}

void FullyConnectedLayer::backwardLastGpu(data_t* expectedOutput) {
    //TODO
}

void FullyConnectedLayer::computeTotalDiffs() {
    
    // COMPUTE TOTAL DERIVATIVES for weights between this and previous layer
    int prevNeurons = previousLayer->getOutputsCount();
    data_t *prevOutputs = previousLayer->getInputs();
    for (int i = 0; i<prevNeurons; i++) {
        for (int j = 0; j<inputsCount; j++) {
            weightDiffs[i*inputsCount+j] = -lr * outputDiffs[j] * prevOutputs[i];
        }
    }

    // COMPUTE BIAS DERIVATIVES for this layer
    if (netConf->getBias()) {
        data_t *biasDiff = weightDiffs + weightsCount - inputsCount;
        for (int i = 0; i<inputsCount; i++) {
            biasDiff[i] = -lr * outputDiffs[i];
        }
    }
    
    // ADJUST WEIGHTS AND BIAS
    for (int i = 0; i<weightsCount; i++) {
        weights[i] += weightDiffs[i];
    }
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
}

static LayerRegister<FullyConnectedLayer> reg("FullyConnected");