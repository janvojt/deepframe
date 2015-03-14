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

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

using namespace std;

template <typename dType>
NetworkConfiguration<dType>::NetworkConfiguration() {
    neuronConf = NULL;
    bias = true;
}

template <typename dType>
NetworkConfiguration<dType>::NetworkConfiguration(const NetworkConfiguration& orig) {
}

template <typename dType>
NetworkConfiguration<dType>::~NetworkConfiguration() {
    if (neuronConf) {
        delete[] neuronConf;
    }
}

template <typename dType>
int NetworkConfiguration<dType>::getLayers() {
    return layers;
}

template <typename dType>
void NetworkConfiguration<dType>::setLayers(int layers) {
    if (layers < 1) {
        throw std::invalid_argument("Number of layers must be a natural number.");
    } else {
        this->layers = layers;
    }
}

template <typename dType>
void NetworkConfiguration<dType>::setNeurons(int layer, int neurons) {
    if (layer > layers || layer < 0) {
        LOG()->error("Provided %d as layer index, which is invalid for network with %d layers.", layer, layers);
        return;
    } else if (neuronConf == NULL) {
        initConf();
    }
    neuronConf[layer] = neurons;
}

template <typename dType>
int NetworkConfiguration<dType>::getNeurons(int layer) {
    return neuronConf[layer];
}

template <typename dType>
void NetworkConfiguration<dType>::setBias(bool enabled) {
    bias = enabled;
}

template <typename dType>
bool NetworkConfiguration<dType>::getBias() {
    return bias;
}

template <typename dType>
void NetworkConfiguration<dType>::initConf() {
    // free memory if it was already assigned to the pointer
    if (neuronConf != NULL) {
        delete neuronConf;
    }
    // initialize configuration
    neuronConf = new int[layers];
}

template <>
void NetworkConfiguration<float>::parseInitInterval(const char* intervalConf) {
    parseInitInterval(intervalConf, "%f");
}

template <>
void NetworkConfiguration<double>::parseInitInterval(const char* intervalConf) {
    parseInitInterval(intervalConf, "%lf");
}

template <typename dType>
void NetworkConfiguration<dType>::parseInitInterval(const char* intervalConf, const char *format) {
    dType v = 0;
    char *haystack = new char[strlen(intervalConf)+1];
    strcpy(haystack, intervalConf);
    char *token = strtok(haystack, ",");
    sscanf(token, format, &v);
    this->initMin = v;
    token = strtok(NULL, ",");
    sscanf(token, format, &v);
    this->initMax = v;
    delete[] haystack;
}

template <typename dType>
void NetworkConfiguration<dType>::parseLayerConf(char* layerConf) {
    
    this->layerConf = layerConf;

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

template <typename dType>
void NetworkConfiguration<dType>::setInitMin(dType min) {
    initMin = min;
}

template <typename dType>
dType NetworkConfiguration<dType>::getInitMin() {
    return initMin;
}

template <typename dType>
void NetworkConfiguration<dType>::setInitMax(dType max) {
    initMax = max;
}

template <typename dType>
dType NetworkConfiguration<dType>::getInitMax() {
    return initMax;
}

template <typename dType>
char* NetworkConfiguration<dType>::getLayerConf() {
    return layerConf;
}

INSTANTIATE_DATA_CLASS(NetworkConfiguration);