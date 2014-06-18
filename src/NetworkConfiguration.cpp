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
    if (layer+1 > layers || layer < 1) {
        LOG()->error("Provided %d as layer index, which is invalid for network with %d layers.", layer, layers);
        return;
    } else if (!neuronConf) {
        initConf(layer);
    }
    neuronConf[layer-1] = neurons;
}

int NetworkConfiguration::getNeurons(int layer) {
    return neuronConf[layer];
}

void NetworkConfiguration::initConf(int layers) {
    // free memory if it was already assigned to the pointer
    if (neuronConf) delete neuronConf;
    // initialize configuration
    int* neuronConf = new int[layers];
    for (int i = 0; i<layers; i++) {
        neuronConf[i] = 0;
    }
}