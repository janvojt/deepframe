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

Network::Network(NetworkConfiguration *conf) {
    this->conf = conf;
    this->noLayers = conf->getLayers();
    this->layers = new Layer*[this->noLayers];
    
    if (conf->getConfSource() != NULL) {
        LOG()->info("Initializing network with layer configuration of (%s).", conf->getConfSource());
    }
}

Network::Network(const Network& orig) {
    this->inputsCount = orig.inputsCount;
    this->weightsCount = orig.weightsCount;
}

Network::~Network() {
}

void Network::setup() {
    if (layerCursor > noLayers) {
        LOG()->error("Network cannot be initialized, because it contains %d out of %d layers.", layerCursor, noLayers);
    } else if (isInitialized) {
        LOG()->warn("Network is already initialized.");
    } else {
        isInitialized = true;
        allocateMemory();
        reinit();
        
        data_t *inputsPtr = inputs;
        data_t *weightsPtr = weights;
        data_t *weightDiffsPtr = weightDiffs;
        for (int i = 0; i<noLayers; i++) {
            Layer *layer = layers[i];
            layer->setInputs(inputsPtr);
            layer->setWeights(weightsPtr, weightDiffsPtr);
            inputsPtr += layer->getOutputsCount();
            weightsPtr += layer->getWeightsCount();
            weightDiffsPtr += layer->getWeightsCount();
        }
    }
}

void Network::forward() {
    if (this->useGpu()) {
        for (int i = 1; i < this->noLayers; i++) {
            LOG()->debug("Computing forward run on GPU for layer %d.", i);
            this->layers[i]->forwardGpu();
        }
    } else {
        for (int i = 1; i < this->noLayers; i++) {
            LOG()->debug("Computing forward run on CPU for layer %d.", i);
            this->layers[i]->forwardCpu();
        }
    }
}

void Network::backward() {
    if (this->useGpu()) {
        for (int i = 1; i < this->noLayers; i++) {
            LOG()->debug("Computing backward run on GPU for layer %d.", i);
            this->layers[i]->backwardGpu();
        }
    } else {
        for (int i = 1; i < this->noLayers; i++) {
            LOG()->debug("Computing backward run on CPU for layer %d.", i);
            this->layers[i]->backwardCpu();
        }
    }
}

void Network::addLayer(Layer* layer) {
    if (layerCursor < noLayers) {
        weightsCount += layer->getWeightsCount();
        inputsCount += layer->getOutputsCount();
        layers[layerCursor++] = layer;
    } else {
        LOG()->error("Cannot add more than %d preconfigured layers.", noLayers);
    }
}

Layer* Network::getLayer(int index) {
    return this->layers[index];
}

NetworkConfiguration* Network::getConfiguration() {
    return this->conf;
}

int Network::getInputNeurons() {
    return this->layers[0]->getOutputsCount();
}

int Network::getOutputNeurons() {
    return this->layers[noLayers-1]->getOutputsCount();
}

data_t *Network::getInputs() {
    return this->inputs;
}

int Network::getInputsCount() {
    return this->inputsCount;
}

data_t* Network::getWeights() {
    return this->weights;
}

int Network::getWeightsCount() {
    return weightsCount;
}

