/* 
 * File:   NetworkConfiguration.cpp
 * Author: Jan Vojt
 * 
 * Created on June 5, 2014, 8:16 PM
 */

#include "NetworkConfiguration.h"

#include <cstdlib>
#include <iostream>
#include <string.h>
#include <stdexcept>

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

using namespace std;

NetworkConfiguration::NetworkConfiguration() {
    neuronConf = NULL;
    bias = true;
}

NetworkConfiguration::NetworkConfiguration(const NetworkConfiguration& orig) {
}

NetworkConfiguration::~NetworkConfiguration() {
    if (neuronConf) {
        delete[] neuronConf;
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

void NetworkConfiguration::setBias(bool enabled) {
    bias = enabled;
}

bool NetworkConfiguration::getBias() {
    return bias;
}

void NetworkConfiguration::initConf() {
    // free memory if it was already assigned to the pointer
    if (neuronConf != NULL) {
        delete neuronConf;
    }
    // initialize configuration
    neuronConf = new int[layers];
}

void NetworkConfiguration::parseLayerConf(char* layerConf) {

    // Configure layers.
    // Count and set number of layers.
    int i;
    char *lconf = layerConf;
    for (i=0; lconf[i]; lconf[i]==',' ? i++ : *lconf++);
    setLayers(i+1);
    
    // set number of neurons for each layer
    i = 0;
    int l = 0;
    char *haystack = new char[strlen(layerConf)+1];
    strcpy(haystack, layerConf);
    char *token = strtok(haystack, ",");
    while (token != NULL) {
        sscanf(token, "%d", &l);
        setNeurons(i++, l);
        token = strtok(NULL, ",");
    }
    delete[] haystack;
}
