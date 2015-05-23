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

INSTANTIATE_DATA_CLASS(Network);