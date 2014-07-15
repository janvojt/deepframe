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
    allocateCache();
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
    computeOutputGradients(expectedOutput);
    computeWeightDifferentials();
    adjustWeights();
}

void BackpropagationLearner::computeOutputGradients(float *expectedOutput) {
    int on = network->getOutputNeurons();
    int noLayers = network->getConfiguration()->getLayers();
    float *localGradient = localGradients + network->getPotentialIndex(noLayers-1);
    float *output = network->getOutput();
    void (*daf) (float*,float*,int) = network->getConfiguration()->dActivationFnc;
    
    // compute local gradients
    float *dv = new float[network->getOutputNeurons()];
    daf(network->getPotentialValues() + network->getPotentialIndex(noLayers-1), dv, on);
    for (int i = 0; i<on; i++) {
        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
    }
}

void BackpropagationLearner::computeWeightDifferentials() {
    int noLayers = network->getConfiguration()->getLayers();
    void (*daf) (float*,float*,int) = network->getConfiguration()->dActivationFnc;
    
    for (int l = noLayers-1; l>0; l--) {
        
        // INITIALIZE HELPER VARIABLES
        int thisPotentialIndex = network->getPotentialIndex(l-1);
        float *thisLocalGradient = localGradients + thisPotentialIndex;
        int nextPotentialIndex = network->getPotentialIndex(l);
        float *nextLocalGradient = localGradients + nextPotentialIndex;
        int thisNeurons = network->getConfiguration()->getNeurons(l-1);
        int nextNeurons = network->getConfiguration()->getNeurons(l);
        float *thisPotential = network->getPotentialValues() + thisPotentialIndex;
        float *weights = network->getWeights() + network->getWeightsIndex(l-1);
        
        
        // COMPUTE TOTAL DERIVATIVES for weights between layer l and l+1
        int wc = network->getWeightsIndex(l+1) - network->getWeightsIndex(l);
        float *thisInputs = network->getInputValues() + network->getPotentialIndex(l-1);
        float *wdiff = weightDiffs + network->getWeightsIndex(l);
        for (int i = 0; i<wc; i++) {
            wdiff[i] = -learningRate * nextLocalGradient[i%nextNeurons] * thisInputs[i/nextNeurons];
        }
        
        
        // COMPUTE LOCAL GRADIENTS for layer l
        
        // compute derivatives of neuron potentials in layer l
        float *thisPotentialDerivatives = new float[thisNeurons];
        daf(thisPotential, thisPotentialDerivatives, thisNeurons);
        
        // compute local gradients for layer l
        for (int i = 0; i<thisNeurons; i++) {
            float sumNextGradient = 0;
            for (int j = 0; j<nextNeurons; j++) {
                sumNextGradient += nextLocalGradient[j] * weights[i * thisNeurons + j];
            }
            *thisLocalGradient = sumNextGradient * thisPotentialDerivatives[i];
        }
    }
}

void BackpropagationLearner::adjustWeights() {
    int wc = network->getWeightsIndex(network->getConfiguration()->getLayers());
    float *weights = network->getWeights();
    // we should skip the garbage in zero-layer weights
    for(int i = network->getWeightsIndex(0); i<wc; i++) {
        weights[i] -= weightDiffs[i];
    }
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
