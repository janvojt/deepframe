/* 
 * File:   Network.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "Network.h"

#include <cstring>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
Network<dType>::Network(NetworkConfiguration<dType> *conf) {
    this->conf = conf;
    this->noLayers = conf->getLayers();
    this->layers = new Layer<dType>*[this->noLayers];
    
    if (conf->getLayerConf() != NULL) {
        LOG()->info("Initializing network with layer configuration of (%s).", conf->getLayerConf());
    }
}

template <typename dType>
Network<dType>::Network(const Network& orig) {
    this->inputsCount = orig.inputsCount;
    this->weightsCount = orig.weightsCount;
}

template <typename dType>
Network<dType>::~Network() {
}

template<typename dType>
void Network<dType>::setup() {
    if (layerCursor > noLayers) {
        LOG()->error("Network cannot be initialized, because it contains %d out of %d layers.", layerCursor, noLayers);
    } else if (isInitialized) {
        LOG()->warn("Network is already initialized.");
    } else {
        isInitialized = true;
        allocateMemory();
        reinit();
        
        dType *inputsPtr = inputs;
        dType *weightsPtr = weights;
        for (int i = 0; i<noLayers; i++) {
            Layer<dType> *layer = layers[i];
            layer->setInputs(inputs);
            layer->setWeights(weights);
            inputsPtr += layer->getOutputCount();
            weightsPtr += layer->getWeightCount();
        }
    }
}

template<typename dType>
void Network<dType>::addLayer(Layer<dType>* layer) {
    if (layerCursor < noLayers) {
        weightsCount += layer->getWeightCount();
        inputsCount += layer->getOutputCount();
        layers[layerCursor++] = layer;
    } else {
        LOG()->error("Cannot add more than %d preconfigured layers.", noLayers);
    }
}

template<typename dType>
Layer<dType>* Network<dType>::getLayer(int index) {
    return this->layers[index];
}

template <typename dType>
NetworkConfiguration<dType>* Network<dType>::getConfiguration() {
    return this->conf;
}

template <typename dType>
int Network<dType>::getInputNeurons() {
    return this->layers[0]->getOutputCount();
}

template <typename dType>
int Network<dType>::getOutputNeurons() {
    return this->layers[noLayers-1]->getOutputCount();
}

template <typename dType>
dType *Network<dType>::getInputs() {
    return this->inputs;
}

template <typename dType>
int Network<dType>::getInputsCount() {
    return this->inputsCount;
}

template <typename dType>
dType* Network<dType>::getWeights() {
    return this->weights;
}

template<typename dType>
int Network<dType>::getWeightsCount() {
    return weightsCount;
}


INSTANTIATE_DATA_CLASS(Network);