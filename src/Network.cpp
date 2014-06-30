/* 
 * File:   Network.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "Network.h"

#include <cstring>

Network::Network(NetworkConfiguration *conf) {
    this->conf = conf;
    this->noLayers = conf->getLayers();
    this->bias = 0;
    initWeights();
    initInputs();
}

Network::Network(const Network& orig) {
}

Network::~Network() {
    delete weights;
    delete inputs;
}

NetworkConfiguration* Network::getConfiguration() {
    return conf;
}

void Network::initWeights() {
    int noWeights = 0;
    int pLayer = 1; // neurons in previous layer
    for (int i = 0; i<noLayers; i++) {
        int tLayer = this->conf->getNeurons(i);
        noWeights += pLayer * tLayer;
        pLayer = tLayer;
    }
    weights = new float[noWeights];
    
    // initialize weights for input layer to 1
    for (int i = 0; i<getInputNeurons(); i++) {
        weights[i] = 1;
    }
}

void Network::initInputs() {
    int noNeurons = 0;
    for (int i = 0; i<noLayers; i++) {
        noNeurons += conf->getNeurons(i);
    }
    this->noNeurons = noNeurons;
    inputs = new float[noNeurons];
}

void Network::setInput(float* input) {
    std::memcpy(inputs, input, sizeof(float) * getInputNeurons());
}

void Network::run() {
    // for every layer
    int nPrevLayers = 0;
    for (int l = 0; l<noLayers; l++) {
        int nThisLayer = conf->getNeurons(l);
        int nNextLayer = conf->getNeurons(l+1);
        // for every neuron in (l)th layer
        for (int i = 0; i<nThisLayer; i++) {
            // for every neuron in (l+1)th layer
            for (int j = 0; j<nNextLayer; j++) {
                int indexFrom = nPrevLayers + i;
                int indexTo = nPrevLayers + nThisLayer + j;
                inputs[indexTo] += inputs[indexFrom];
            }
        }
        nPrevLayers += nThisLayer;
    }
}

float* Network::getOutput() {
    return inputs + noNeurons - getOutputNeurons();
}

int Network::getInputNeurons() {
    return conf->getNeurons(0);
}

int Network::getOutputNeurons() {
    return conf->getNeurons(noLayers-1);
}
