/* 
 * File:   Layer.cpp
 * Author: janvojt
 * 
 * Created on May 16, 2015, 2:19 PM
 */

#include "Layer.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

Layer::Layer() {
}

Layer::Layer(const Layer& orig) {
}

Layer::~Layer() {
}

void Layer::setup(Layer* previousLayer, NetworkConfiguration* netConf, string confString) {
    this->previousLayer = previousLayer;
    if (previousLayer != NULL) {
        previousLayer->setNextLayer(this);
    }
    this->netConf = netConf;
    this->setup(confString);
}

data_t* Layer::getInputs() {
    return this->inputs;
}

void Layer::setInputs(data_t* inputs) {
    this->inputs = inputs;
}

data_t* Layer::getWeights() {
    return this->weights;
}

void Layer::setWeights(data_t* weights, data_t *weightDiffs) {
    this->weights = weights;
    this->weightDiffs = weightDiffs;
}

void Layer::setNextLayer(Layer* nextLayer) {
    this->nextLayer = nextLayer;
    this->isLast = false;
}

int Layer::getWeightsCount() {
    return weightsCount;
}

int Layer::getOutputsCount() {
    return inputsCount;
}
