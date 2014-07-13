/* 
 * File:   BackpropeagationLearner.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 12:10 AM
 */

#include "BackpropagationLearner.h"
#include "Network.h"
#include <cstring>
#include <string>


BackpropagationLearner::BackpropagationLearner(Network *network) {
    this->network = network;
    learningRate = 1;
    epochCounter = 0;
    errorTotal = std::numeric_limits<float>::infinity();
    initPotentials();
    initInputs();
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    
}

void BackpropagationLearner::learn() {
    epochCounter++;
    // TODO
}


void BackpropagationLearner::initInputs() {
    inputs = new float[network->getAllNeurons()];
}

void BackpropagationLearner::initPotentials() {
    potentials = new float[network->getAllNeurons()];
}

void BackpropagationLearner::clearLayer(float *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}
