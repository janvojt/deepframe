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
        this->first = false;
    }
    this->netConf = netConf;
    this->lr = netConf->getLearningRate();
    this->setup(confString);
}

data_t* Layer::getOutputs() {
    return this->outputs;
}

data_t* Layer::getOutputDiffs() {
    return this->outputDiffs;
}

void Layer::setInputs(data_t* inputs, data_t *outputDiffs) {
    this->outputs = inputs;
    this->outputDiffs = outputDiffs;
}

data_t* Layer::getWeights() {
    return this->weights;
}

data_t* Layer::getWeightDiffs() {
    return this->weightDiffs;
}

void Layer::setWeights(data_t* weights, data_t *weightDiffs) {
    this->weights = weights;
    this->weightDiffs = weightDiffs;
}

void Layer::setNextLayer(Layer* nextLayer) {
    this->nextLayer = nextLayer;
    this->last = false;
}

int Layer::getWeightsCount() {
    return weightsCount;
}

int Layer::getOutputsCount() {
    return outputsCount;
}

bool Layer::isFirst() {
    return first;
}

bool Layer::isLast() {
    return last;
}
