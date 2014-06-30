/* 
 * File:   NetworkConfiguration.cpp
 * Author: Jan Vojt
 * 
 * Created on June 5, 2014, 8:16 PM
 */

#include "NetworkConfiguration.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

NetworkConfiguration::NetworkConfiguration() {
    neuronConf = NULL;
}

NetworkConfiguration::NetworkConfiguration(const NetworkConfiguration& orig) {
}

NetworkConfiguration::~NetworkConfiguration() {
    if (neuronConf) {
        delete neuronConf;
    }
}

int NetworkConfiguration::getLayers() {
    return layers;
}

void NetworkConfiguration::setLayers(int layers) {
    if (layers < 1) {
        throw std::invalid_argument("Number of layers must be a natural number.");
    } else {
        this->layers = layers;
    }
}


void NetworkConfiguration::setNeurons(int layer, int neurons) {
    if (layer > layers || layer < 0) {
        LOG()->error("Provided %d as layer index, which is invalid for network with %d layers.", layer, layers);
        return;
    } else if (neuronConf == NULL) {
        initConf();
    }
    neuronConf[layer] = neurons;
}

int NetworkConfiguration::getNeurons(int layer) {
    return neuronConf[layer];
}

void NetworkConfiguration::initConf() {
    // free memory if it was already assigned to the pointer
    if (neuronConf != NULL) {
        delete neuronConf;
    }
    // initialize configuration
    neuronConf = new int[layers];
}