/* 
 * File:   Network.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "Network.h"

#include <cstring>
#include <string>

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
    for (int i = 0; i<noLayers; i++) {
        int tLayer = this->conf->getNeurons(i);
        noWeights += pLayer * tLayer;
        weightsUpToLayerCache[i] = noWeights;
        pLayer = tLayer;
    }
    weights = new double[noWeights];
    
    // initialize weights to 1
    std::fill_n(weights, noWeights, 1);
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
    potentials = new double[noNeurons];
    inputs = new double[noNeurons];
}

void Network::initBias() {
    if (conf->getBias()) {
        bias = new double[noNeurons];
        std::fill_n(bias, noNeurons, 1);
    } else {
        bias = NULL;
    }
}

void Network::run() {
    // number of neurons in so far processed layers
    int nPrevLayers = 0;
    double *weighPtr = weights + getInputNeurons();
    
    // first normalize inputs using activation function
    conf->activationFnc(potentials, inputs, getInputNeurons());
    
    // for every layer
    for (int l = 0; l<noLayers-1; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        
        // clear the following layer just before working with it
        clearLayer(potentials + nPrevLayers + nThisLayer, nNextLayer);
        clearLayer(inputs + nPrevLayers + nThisLayer, nNextLayer);
        
        // for every neuron in (l)th layer
        for (int i = 0; i<nThisLayer; i++) {
            int indexFrom = nPrevLayers + i;
            // for every neuron in (l+1)th layer
            for (int j = 0; j<nNextLayer; j++) {
                int indexTo = nPrevLayers + nThisLayer + j;
                potentials[indexTo] += *weighPtr * inputs[indexFrom];
                weighPtr++;
            }
        }

        applyBias(l+1);
        
        // Run through activation function
        conf->activationFnc(potentials+nPrevLayers+nThisLayer, inputs+nPrevLayers+nThisLayer, nNextLayer);
        
        nPrevLayers += nThisLayer;
    }
}

void Network::applyBias(int l) {
    int n = conf->getNeurons(l);
    int potentialOffset = getPotentialOffset(l);
    for (int i = 0; i<n; i++) {
        potentials[potentialOffset + i] += bias[potentialOffset + i];
    }
}

void Network::clearLayer(double *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}

void Network::setInput(double* input) {
    std::memcpy(potentials, input, sizeof(double) * getInputNeurons());
}

double *Network::getInput() {
    return potentials;
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

double* Network::getPotentialValues() {
    return potentials;
}

int Network::getPotentialOffset(int layer) {
    return neuronsUpToLayerCache[layer];
}

double* Network::getInputValues() {
    return inputs;
}

double* Network::getWeights() {
    return weights;
}

int Network::getWeightsOffset(int layer) {
    return weightsUpToLayerCache[layer-1];
}

double* Network::getBiasValues() {
    return bias;
}
