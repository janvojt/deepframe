/* 
 * File:   FullyConnectedLayer.cpp
 * Author: janvojt
 * 
 * Created on May 17, 2015, 12:55 AM
 */

#include "FullyConnectedLayer.h"

#include <algorithm>
#include "../../common.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template<typename dType>
FullyConnectedLayer<dType>::FullyConnectedLayer() {
}

template<typename dType>
FullyConnectedLayer<dType>::FullyConnectedLayer(const FullyConnectedLayer& orig) {
}

template<typename dType>
FullyConnectedLayer<dType>::~FullyConnectedLayer() {
}

template<typename dType>
void FullyConnectedLayer<dType>::setup(Layer<dType> *previousLayer, FullyConnectedConfig<dType> conf) {
    this->conf = conf;
    if (previousLayer != NULL) {
        // this is not the input layer
        this->previousLayer = previousLayer;
        this->weightSize = previousLayer->getOutputCount() * conf.outputSize;
        if (conf.useBias) {
            this->weightSize += conf.outputSize;
        }
        previousLayer->setNextLayer(this);
    }
}

template<typename dType>
void FullyConnectedLayer<dType>::forward() {
    
    int inputSize = this->previousLayer->getOutputCount();
    dType *inputPtr = this->previousLayer->getInputs();
    dType *outputPtr = this->getInputs();
    
    // Clear output neurons
    std::fill_n(outputPtr, conf.outputSize, 0);
    
    dType *weightPtr = this->weights;
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

template<typename dType>
int FullyConnectedLayer<dType>::getWeightCount() {
    return weightSize;
}

template<typename dType>
int FullyConnectedLayer<dType>::getOutputCount() {
    return conf.outputSize;
}

INSTANTIATE_DATA_CLASS(FullyConnectedLayer);