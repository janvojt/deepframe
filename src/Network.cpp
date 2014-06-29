/* 
 * File:   Network.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "Network.h"

Network::Network(NetworkConfiguration *conf) {
    this->conf = conf;
    initWeights();
    initInputs();
    bias = 0;
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
    int pLayer = 1;
    for (int i = 1; i<=noLayers; i++) {
        noWeights += pLayer *this->conf->getNeurons(i);
    }
    weights = new float[noWeights];
    
    // initialize weights for input layer to 1
    for (int i = 0; i<conf->getNeurons(1); i++) {
        weights[i] = 1;
    }
}

void Network::initInputs() {
    int noNeurons = 0;
    for (int i = 1; i<=noLayers; i++) {
        noNeurons += conf->getNeurons(i);
    }
    inputs = new float[noNeurons];
}

void Network::setInput(float* input) {
    // TODO use memcpy instead of for loop
    for (int i = 0; i<conf->getNeurons(1); i++) {
        inputs[i] = input[i];
    }
}
