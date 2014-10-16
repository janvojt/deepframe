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

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

Network::Network(NetworkConfiguration *conf) {
    this->conf = conf;
    this->noLayers = conf->getLayers();
    initWeights();
    initInputs();
    initBias();
}

Network::Network(const Network& orig) {
}

Network::~Network() {
    delete weightsUpToLayerCache;
    delete neuronsUpToLayerCache;
    delete weights;
    delete inputs;
    delete bias;
}

NetworkConfiguration* Network::getConfiguration() {
    return conf;
}

void Network::initWeights() {
    int noWeights = 0;
    int pLayer = 1; // neurons in previous layer
    weightsUpToLayerCache = new int[noLayers];
    weightsUpToLayerCache[0] = noWeights;
    for (int i = 0; i<noLayers; i++) {
        int tLayer = this->conf->getNeurons(i);
        noWeights += pLayer * tLayer;
        weightsUpToLayerCache[i+1] = noWeights;
        pLayer = tLayer;
    }
    weights = new double[noWeights];

    // Initialize weights.
    if (conf->getInitRandom()) {
        LOG()->info("Randomly initialize weights between -1 and 1.");
        for (int i = 0; i < noWeights; i++) {
            weights[i] = (double) (rand()) / (RAND_MAX / 2) - 1;
        }
    } else {
        LOG()->info("Initialize all weights to a constant value of %f.", conf->getInitWeights());
        std::fill_n(weights, noWeights, conf->getInitWeights());
    }
}

void Network::initInputs() {
    int noNeurons = 0;
    neuronsUpToLayerCache = new int[noLayers];
    neuronsUpToLayerCache[0] = noNeurons;
    for (int i = 0; i<noLayers; i++) {
        noNeurons += conf->getNeurons(i);
        neuronsUpToLayerCache[i+1] = noNeurons;
    }
    this->noNeurons = noNeurons;
    inputs = new double[noNeurons];
}

void Network::initBias() {
    if (conf->getBias()) {
        bias = new double[noNeurons];
        
        // Initialize bias.
        if (conf->getInitRandom()) {
            // Randomly initialize bias between -1 and 1.
            for (int i = 0; i < noNeurons; i++) {
                bias[i] = (double) (rand()) / (RAND_MAX / 2) - 1;
            }
        } else {
            // Initialize all bias to a constant.
            std::fill_n(weights, noNeurons, conf->getInitWeights());
        }
    } else {
        bias = NULL;
    }
}

void Network::run() {
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
    
    LOG()->debug("Forward phase results: [%f, %f] -> [%f].", getInput()[0], getInput()[1], getOutput()[0]);
}

void Network::applyBias(int l) {
    int n = conf->getNeurons(l);
    int offset = getInputOffset(l);
    for (int i = 0; i<n; i++) {
        inputs[offset + i] += bias[offset + i];
    }
}

void Network::clearLayer(double *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}

void Network::setInput(double* input) {
    std::memcpy(inputs, input, sizeof(double) * getInputNeurons());
}

double *Network::getInput() {
    return inputs;
}

double *Network::getOutput() {
    return inputs + noNeurons - getOutputNeurons();
}

int Network::getInputNeurons() {
    return conf->getNeurons(0);
}

int Network::getOutputNeurons() {
    return conf->getNeurons(noLayers-1);
}

int Network::getAllNeurons() {
    return noNeurons;
}

int Network::getInputOffset(int layer) {
    return neuronsUpToLayerCache[layer];
}

double* Network::getWeights() {
    return weights;
}

int Network::getWeightsOffset(int layer) {
    return weightsUpToLayerCache[layer];
}

double* Network::getBiasValues() {
    return bias;
}
