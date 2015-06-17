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
#include "../../util/cpuDebugHelpers.h"

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
    this->outputsCount = conf.outputSize;
    LOG()->debug("Fully connected layer size is %d neurons.", this->outputsCount);
}

void FullyConnectedLayer::forwardCpu() {
    
    int inputSize = this->previousLayer->getOutputsCount();
    data_t *inputPtr = this->previousLayer->getOutputs();
    data_t *outputPtr = this->getOutputs();
    
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
//            LOG()->debug("Input %d after applying bias: %f.", i, outputPtr[i]);
            weightPtr++;
        }
    }
    
    // Run through activation function
    netConf->activationFnc(outputPtr, outputPtr, conf.outputSize);
//    dumpHostArray('O', outputPtr, outputsCount);
}

void FullyConnectedLayer::forwardGpu() {
    //TODO
}


void FullyConnectedLayer::backwardCpu() {
        
    // COMPUTE LOCAL GRADIENTS for this layer

    // compute derivatives of neuron inputs for this layer
    void (*daf) (data_t*,data_t*,int) = this->netConf->dActivationFnc;
    data_t *thisInputDerivatives = new data_t[outputsCount];
    daf(outputs, thisInputDerivatives, outputsCount);

    // compute local gradients for this layer
    int nextNeurons = nextLayer->getOutputsCount();
    data_t *nextOutputDiffs = nextLayer->getOutputDiffs();
    data_t *nextWeights = nextLayer->getWeights();
    for (int i = 0; i<outputsCount; i++) {
        data_t sumNextGradient = 0;
        for (int j = 0; j<nextNeurons; j++) {
            sumNextGradient += nextOutputDiffs[j] * nextWeights[i * nextNeurons + j];
        }
        outputDiffs[i] = sumNextGradient * thisInputDerivatives[i];
//            LOG()->debug("Local gradient for neuron [%d, %d] : %f.", l, i, thisLocalGradient[i]);
    }
    delete[] thisInputDerivatives;
//    dumpHostArray('l', outputDiffs, outputsCount);
    
    computeTotalDiffs();
}

void FullyConnectedLayer::backwardLastCpu(data_t* expectedOutput) {
    
//    LOG()->debug("Backpropagating (%f, %f) -> (%f).", *(outputs-4), *(outputs-3), *expectedOutput);
    
    void (*daf) (data_t*,data_t*,int) = this->netConf->dActivationFnc;
    
    // compute local gradients
    data_t *dv = new data_t[outputsCount]; //TODO allocate only once
    daf(outputs, dv, outputsCount);
    for (int i = 0; i<outputsCount; i++) {
        outputDiffs[i] = (outputs[i] - expectedOutput[i]) * dv[i];
    }
    delete[] dv;
//    dumpHostArray('o', outputDiffs, outputsCount);
}

void FullyConnectedLayer::backwardGpu() {
    //TODO
}

void FullyConnectedLayer::backwardLastGpu(data_t* expectedOutput) {
    //TODO
}

void FullyConnectedLayer::computeTotalDiffs() {
    
    // Initialize helper variables
    int nextNeurons = nextLayer->getOutputsCount();
    int nextWeightsCount = nextLayer->getWeightsCount();
    data_t *nextWeights = nextLayer->getWeights();
    data_t *nextWeightDiffs = nextLayer->getWeightDiffs();
    data_t *nextOutputDiffs = nextLayer->getOutputDiffs();
    
    // COMPUTE TOTAL DERIVATIVES for weights between this and next layer
    for (int i = 0; i<outputsCount; i++) {
        for (int j = 0; j<nextNeurons; j++) {
            nextWeightDiffs[i*nextNeurons+j] = -lr * nextOutputDiffs[j] * outputs[i];
        }
    }
//    dumpHostArray('w', nextWeightDiffs, nextNeurons * outputsCount);

    // COMPUTE BIAS DERIVATIVES for next layer
    if (netConf->getBias()) {
        data_t *nextBiasDiff = nextWeightDiffs + nextWeightsCount - nextNeurons;
        for (int i = 0; i<nextNeurons; i++) {
            nextBiasDiff[i] = -lr * nextOutputDiffs[i];
        }
//        dumpHostArray('b', nextBiasDiff, nextNeurons);
    }
    
    // ADJUST WEIGHTS AND BIAS in next layer
    for (int i = 0; i<nextWeightsCount; i++) {
        nextWeights[i] += nextWeightDiffs[i];
    }
//    dumpHostArray('W', nextWeights, nextWeightsCount-nextNeurons);
//    dumpHostArray('B', nextWeights+nextWeightsCount-nextNeurons, nextNeurons);
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