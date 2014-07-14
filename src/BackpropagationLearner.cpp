/* 
 * File:   BackpropeagationLearner.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 12:10 AM
 */

#include "BackpropagationLearner.h"
#include "Network.h"
#include "LabeledDataset.h"
#include <cstring>
#include <string>
#include <stdexcept>


BackpropagationLearner::BackpropagationLearner(Network *network) {
    this->network = network;
    learningRate = 1;
    epochCounter = 0;
    errorTotal = std::numeric_limits<float>::infinity();
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    
}

void BackpropagationLearner::train(LabeledDataset *dataset) {
    float errorPrev = 0;
    do {
        epochCounter++;
        while (dataset->hasNext()) {
            float *pattern = dataset->next();
            doForwardPhase(pattern);
            doBackwardPhase(pattern + dataset->getInputDimension());
        }
    } while (errorTotal < errorPrev);
}

void BackpropagationLearner::doForwardPhase(float* input) {
    network->setInput(input);
    network->run();
}

void BackpropagationLearner::doBackwardPhase(float* output) {
    // TODO
}

void BackpropagationLearner::clearLayer(float *inputPtr, int layerSize) {
    std::fill_n(inputPtr, layerSize, 0);
}

void BackpropagationLearner::validate(LabeledDataset *dataset) {
    if (dataset->getInputDimension() != network->getInputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same input dimension as the number of input neurons!");
    }
    if (dataset->getOutputDimension() != network->getOutputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same output dimension as the number of output neurons!");
    }
}
