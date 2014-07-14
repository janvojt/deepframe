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

void BackpropagationLearner::allocateCache() {
    weightDiffs = new float[network->getWeightsIndex(network->getConfiguration()->getLayers())];
    localGradients = new float[network->getAllNeurons()];
}

void BackpropagationLearner::train(LabeledDataset *dataset) {
    float errorPrev = 0;
    do {
        epochCounter++;
        while (dataset->hasNext()) {
            float *pattern = dataset->next();
            float *output = pattern + dataset->getInputDimension();
            doForwardPhase(pattern);
            doBackwardPhase(output);
        }
    } while (errorTotal < errorPrev); // TODO not applicable without validation set
}

void BackpropagationLearner::doForwardPhase(float *input) {
    network->setInput(input);
    network->run();
}

void BackpropagationLearner::doBackwardPhase(float *expectedOutput) {
    computeOutputLayer(expectedOutput);
    computeHiddenLayers();
    adjustWeights();
}

void BackpropagationLearner::computeOutputLayer(float *expectedOutput) {
    int on = network->getOutputNeurons();
    int noLayers = network->getConfiguration()->getLayers();
    float *localGradient = localGradients + network->getPotentialIndex(noLayers);
    float *output = network->getOutput();
    void (*daf) (float*,float*,int) = network->getConfiguration()->dActivationFnc;
    
    // compute local gradients
    float *dv = new float[network->getOutputNeurons()];
    daf(network->getPotentialValues() + network->getPotentialIndex(noLayers), dv, on);
    for (int i = 0; i<on; i++) {
        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
    }
    
    // compute total differential for weights
    int wc = network->getWeightsIndex(noLayers) - network->getWeightsIndex(noLayers-1);
    float *inputs = network->getInputValues() + network->getPotentialIndex(noLayers-1);
    float *wdiff = weightDiffs + network->getWeightsIndex(noLayers-1);
    for (int i = 0; i<wc; i++) {
        wdiff[i] = -learningRate * localGradient[i%on] * inputs[i/on];
    }
}

void BackpropagationLearner::computeHiddenLayers() {
    //TODO
}

void BackpropagationLearner::adjustWeights() {
    //TODO
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
