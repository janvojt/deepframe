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

void Layer::forward() {
    return this->useGpu ? this->forwardGpu() : this->forwardCpu();
}

void Layer::backward() {
    return this->useGpu ? this->backwardGpu() : this->backwardCpu();
}

void Layer::setUseGpu(bool useGpu) {
    this->useGpu = useGpu;
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

void Layer::setWeights(data_t* weights) {
    this->weights = weights;
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
