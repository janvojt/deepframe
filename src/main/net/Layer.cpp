/* 
 * File:   Layer.cpp
 * Author: janvojt
 * 
 * Created on May 16, 2015, 2:19 PM
 */

#include "Layer.h"

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
Layer<dType>::Layer() {
}

template <typename dType>
Layer<dType>::Layer(const Layer& orig) {
}

template <typename dType>
Layer<dType>::~Layer() {
}

template<typename dType>
dType* Layer<dType>::getInputs() {
    return this->inputs;
}

template<typename dType>
void Layer<dType>::setInputs(dType* inputs) {
    this->inputs = inputs;
}

template<typename dType>
dType* Layer<dType>::getWeights() {
    return this->weights;
}

template<typename dType>
void Layer<dType>::setWeights(dType* weights) {
    this->weights = weights;
}

template<typename dType>
void Layer<dType>::setNextLayer(Layer* nextLayer) {
    this->nextLayer = nextLayer;
    this->isLast = false;
}

template<typename dType>
int Layer<dType>::getWeightsCount() {
    return weightsCount;
}

template<typename dType>
int Layer<dType>::getOutputsCount() {
    return inputsCount;
}

INSTANTIATE_DATA_CLASS(Layer);