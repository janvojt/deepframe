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

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

BackpropagationLearner::BackpropagationLearner(Network *network) {
    this->network = network;
    learningRate = 1;
    epochCounter = 0;
    epochLimit = 1000000;
    targetMse = .0001;
    errorTotal = std::numeric_limits<float>::infinity();
    allocateCache();
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    
}

void BackpropagationLearner::allocateCache() {
    weightDiffs = new float[network->getWeightsOffset(network->getConfiguration()->getLayers())];
    localGradients = new float[network->getAllNeurons()];
}

void BackpropagationLearner::train(LabeledDataset *dataset) {
    float mse;
    LOG()->info("Started training with limits of %d epochs and target MSE of %f.", epochLimit, targetMse);
    do {
        epochCounter++;
        LOG()->debug("Starting epoch %d.", epochCounter);
        dataset->reset();
        int datasetSize = 0;
        mse = 0;
        while (dataset->hasNext()) {
            datasetSize++;
            float *pattern = dataset->next();
            float *expOutput = pattern + dataset->getInputDimension();
            LOG()->debug("Learning pattern [%f, %f] -> [%f].", pattern[0], pattern[1], expOutput[0]);
            doForwardPhase(pattern);
            doBackwardPhase(expOutput);
            mse += errorComputer->compute(network, expOutput);
        }
        mse = mse / datasetSize;
        LOG()->debug("Finished epoch %d with MSE: %f.", epochCounter, mse);
    } while (mse > targetMse && epochCounter < epochLimit);
    LOG()->info("Finished training after %d epochs with MSE of %f.", epochCounter, mse);
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
    float *localGradient = localGradients + network->getPotentialOffset(noLayers-1);
    float *output = network->getOutput();
    void (*daf) (float*,float*,int) = network->getConfiguration()->dActivationFnc;
    
    // compute local gradients
    float *dv = new float[network->getOutputNeurons()];
    daf(network->getPotentialValues() + network->getPotentialOffset(noLayers-1), dv, on);
    for (int i = 0; i<on; i++) {
        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
    }
}

void BackpropagationLearner::computeWeightDifferentials() {
    int noLayers = network->getConfiguration()->getLayers();
    void (*daf) (float*,float*,int) = network->getConfiguration()->dActivationFnc;
    
    for (int l = noLayers-1; l>0; l--) {
        
        // INITIALIZE HELPER VARIABLES
        int thisPotentialIndex = network->getPotentialOffset(l-1);
        float *thisLocalGradient = localGradients + thisPotentialIndex;
        int nextPotentialIndex = network->getPotentialOffset(l);
        float *nextLocalGradient = localGradients + nextPotentialIndex;
        int thisNeurons = network->getConfiguration()->getNeurons(l-1);
        int nextNeurons = network->getConfiguration()->getNeurons(l);
        float *thisPotential = network->getPotentialValues() + thisPotentialIndex;
        float *weights = network->getWeights() + network->getWeightsOffset(l-1);
        
        
        // COMPUTE TOTAL DERIVATIVES for weights between layer l and l+1
        int wc = network->getWeightsOffset(l+1) - network->getWeightsOffset(l);
        float *thisInputs = network->getInputValues() + network->getPotentialOffset(l-1);
        float *wdiff = weightDiffs + network->getWeightsOffset(l);
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
            thisLocalGradient[i] = sumNextGradient * thisPotentialDerivatives[i];
        }
    }
}

void BackpropagationLearner::adjustWeights() {
    int wc = network->getWeightsOffset(network->getConfiguration()->getLayers());
    float *weights = network->getWeights();
    LOG()->debug("Adjusting weights by: [[%f, %f, %f, %f], [%f, %f]].", weightDiffs[2], weightDiffs[3], weightDiffs[4], weightDiffs[5], weightDiffs[6], weightDiffs[7]);
    // we should skip the garbage in zero-layer weights
    for(int i = network->getWeightsOffset(1); i<wc; i++) {
        weights[i] += weightDiffs[i];
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

void BackpropagationLearner::setEpochLimit(int limit) {
    epochLimit = limit;
}

void BackpropagationLearner::setErrorComputer(ErrorComputer* errorComputer) {
    this->errorComputer = errorComputer;
}

void BackpropagationLearner::setTargetMse(float mse) {
    targetMse = mse;
}
