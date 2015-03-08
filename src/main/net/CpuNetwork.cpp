/* 
 * File:   CpuNetwork.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "CpuNetwork.h"

#include <cstring>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "../util/cpuDebugHelpers.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

CpuNetwork::CpuNetwork(NetworkConfiguration *conf) : Network(conf) {
    initWeights();
    initInputs();
    initBias();
    reinit();
}

CpuNetwork::CpuNetwork(const CpuNetwork& orig) : Network(orig.conf) {
    initWeights();
    initInputs();
    initBias();
    
    std::memcpy(inputs, orig.inputs, sizeof(double) * noNeurons);
    std::memcpy(weights, orig.weights, sizeof(double) * weightsUpToLayerCache[noLayers]);
    std::memcpy(bias, orig.bias, sizeof(double) * getInputNeurons());
}

CpuNetwork::~CpuNetwork() {
    delete[] weightsUpToLayerCache;
    delete[] neuronsUpToLayerCache;
    delete[] weights;
    delete[] inputs;
    delete[] bias;
}

CpuNetwork* CpuNetwork::clone() {
    return new CpuNetwork(*this);
}

void CpuNetwork::merge(Network** nets, int size) {
    
    int noWeights = weightsUpToLayerCache[noLayers];
    for (int i = 0; i<size; i++) {
        
        // add weights
        double *oWeights = nets[i]->getWeights();
        for (int j = 0; j<noWeights; j++) {
            weights[j] += oWeights[j];
        }
        
        // add bias
        double *oBias = nets[i]->getBiasValues();
        for (int j = 0; j<noNeurons; j++) {
            bias[j] += oBias[j];
        }
    }
    
    // divide to get the average
    for (int j = 0; j<noWeights; j++) {
        weights[j] /= size+1;
    }
    for (int j = 0; j<noNeurons; j++) {
        bias[j] /= size+1;
    }
}

void CpuNetwork::reinit() {
    
    LOG()->info("Randomly initializing weights within the interval (%f,%f).", conf->getInitMin(), conf->getInitMax());
    
    // overwrite weights with random doubles
    randomizeDoubles(&weights, weightsUpToLayerCache[noLayers]);
    
    if (conf->getBias()) {
    
        LOG()->info("Randomly initializing bias within the interval (%f,%f).", conf->getInitMin(), conf->getInitMax());
        if (bias == NULL) {
            initBias();
        }
        
        // overwrite bias with random doubles
        randomizeDoubles(&bias, noNeurons);
    }
}


void CpuNetwork::initWeights() {
    int noWeights = 0;
    int pLayer = 1; // neurons in previous layer
    weightsUpToLayerCache = new int[noLayers+1];
    weightsUpToLayerCache[0] = noWeights;
    for (int i = 0; i<noLayers; i++) {
        int tLayer = this->conf->getNeurons(i);
        noWeights += pLayer * tLayer;
        weightsUpToLayerCache[i+1] = noWeights;
        pLayer = tLayer;
    }
}

void CpuNetwork::initInputs() {
    int noNeurons = 0;
    neuronsUpToLayerCache = new int[noLayers+1];
    neuronsUpToLayerCache[0] = noNeurons;
    for (int i = 0; i<noLayers; i++) {
        noNeurons += conf->getNeurons(i);
        neuronsUpToLayerCache[i+1] = noNeurons;
    }
    this->noNeurons = noNeurons;
    inputs = new double[noNeurons];
}

void CpuNetwork::initBias() {
    if (conf->getBias()) {
        bias = new double[noNeurons];
    } else {
        bias = NULL;
    }
}

void CpuNetwork::run() {
    // number of neurons in so far processed layers
    int nPrevLayers = 0;
    double *weighPtr = weights + getInputNeurons();
    
    // for every layer
    for (int l = 0; l<noLayers-1; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        clearLayer(inputs + nPrevLayers + nThisLayer, nNextLayer);
        
        // for every neuron in (l)th layer
        for (int i = 0; i<nThisLayer; i++) {
            int indexFrom = nPrevLayers + i;
            // for every neuron in (l+1)th layer
            for (int j = 0; j<nNextLayer; j++) {
                int indexTo = nPrevLayers + nThisLayer + j;
                inputs[indexTo] += *weighPtr * inputs[indexFrom];
                weighPtr++;
            }
        }

        if (conf->getBias()) {
            applyBias(l+1);
        }
        
        // Run through activation function
        conf->activationFnc(inputs+nPrevLayers+nThisLayer, inputs+nPrevLayers+nThisLayer, nNextLayer);
        
        nPrevLayers += nThisLayer;
    }
}

void CpuNetwork::randomizeDoubles(double** memPtr, int size) {
    *memPtr = new double[size];
    double *mem = *memPtr;
    double min = conf->getInitMin();
    double max = conf->getInitMax();
    double interval = max - min;
    for (int i = 0; i < size; i++) {
        mem[i] = ((double) (rand()) / RAND_MAX * interval) + min;
    }
}

void CpuNetwork::applyBias(int l) {
    int n = conf->getNeurons(l);
    int offset = getInputOffset(l);
    for (int i = 0; i<n; i++) {
        inputs[offset + i] += bias[offset + i];
    }
}

void CpuNetwork::clearLayer(double *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}

void CpuNetwork::setInput(double* input) {
    std::memcpy(inputs, input, sizeof(double) * getInputNeurons());
}

double *CpuNetwork::getInputs() {
    return inputs;
}

double *CpuNetwork::getInput() {
    return inputs;
}

double *CpuNetwork::getOutput() {
    return inputs + neuronsUpToLayerCache[noLayers-1];
}

int CpuNetwork::getAllNeurons() {
    return noNeurons;
}

int CpuNetwork::getInputOffset(int layer) {
    return neuronsUpToLayerCache[layer];
}

double* CpuNetwork::getWeights() {
    return weights;
}

int CpuNetwork::getWeightsOffset(int layer) {
    return weightsUpToLayerCache[layer];
}

double* CpuNetwork::getBiasValues() {
    return bias;
}
