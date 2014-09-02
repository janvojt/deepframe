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
    errorTotal = std::numeric_limits<double>::infinity();
    allocateCache();
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    
}

void BackpropagationLearner::allocateCache() {
    weightDiffs = new double[network->getWeightsOffset(network->getConfiguration()->getLayers())];
    localGradients = new double[network->getAllNeurons()];
    
    useBias = network->getConfiguration()->getBias();
    biasDiff = useBias ? new double[network->getAllNeurons()] : NULL;
}

void BackpropagationLearner::train(LabeledDataset *dataset) {
    double mse;
    LOG()->info("Started training with limits of %d epochs and target MSE of %f.", epochLimit, targetMse);
    do {
        epochCounter++;
        LOG()->debug("Starting epoch %d.", epochCounter);
        dataset->reset();
        int datasetSize = 0;
        mse = 0;
        while (dataset->hasNext()) {
            datasetSize++;
            double *pattern = dataset->next();
            double *expOutput = pattern + dataset->getInputDimension();
            LOG()->debug("Learning pattern [%f, %f] -> [%f].", pattern[0], pattern[1], expOutput[0]);
            doForwardPhase(pattern);
            doBackwardPhase(expOutput);
            mse += errorComputer->compute(network, expOutput);
        }
        mse = mse / datasetSize;
        LOG()->debug("Finished epoch %d with MSE: %f.", epochCounter, mse);
    } while (mse > targetMse && epochCounter < epochLimit);
    
    if (mse <= targetMse) {
        LOG()->info("Training successful after %d epochs with MSE of %f.", epochCounter, mse);
    } else {
        LOG()->info("Training interrupted after %d epochs with MSE of %f.", epochCounter, mse);
    }
}

void BackpropagationLearner::doForwardPhase(double *input) {
    network->setInput(input);
    network->run();
}

void BackpropagationLearner::doBackwardPhase(double *expectedOutput) {
    computeOutputGradients(expectedOutput);
    computeWeightDifferentials();
    adjustWeights();
    if (network->getConfiguration()->getBias()) {
        adjustBias();
    }
}

void BackpropagationLearner::computeOutputGradients(double *expectedOutput) {
    int on = network->getOutputNeurons();
    int noLayers = network->getConfiguration()->getLayers();
    double *localGradient = localGradients + network->getInputOffset(noLayers-1);
    double *output = network->getOutput();
    void (*daf) (double*,double*,int) = network->getConfiguration()->dActivationFnc;
    
    // compute local gradients
    double *dv = new double[network->getOutputNeurons()];
    daf(network->getInput() + network->getInputOffset(noLayers-1), dv, on);
    for (int i = 0; i<on; i++) {
        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
    }
}

void BackpropagationLearner::computeWeightDifferentials() {
    int noLayers = network->getConfiguration()->getLayers();
    void (*daf) (double*,double*,int) = network->getConfiguration()->dActivationFnc;
    
    for (int l = noLayers-1; l>0; l--) {
        
        // INITIALIZE HELPER VARIABLES
        int thisInputIdx = network->getInputOffset(l-1);
        double *thisLocalGradient = localGradients + thisInputIdx;
        int nextInputIdx = network->getInputOffset(l);
        double *nextLocalGradient = localGradients + nextInputIdx;
        int thisNeurons = network->getConfiguration()->getNeurons(l-1);
        int nextNeurons = network->getConfiguration()->getNeurons(l);
        double *thisInput = network->getInput() + thisInputIdx;
        double *weights = network->getWeights() + network->getWeightsOffset(l-1);
        
        
        // COMPUTE TOTAL DERIVATIVES for weights between layer l and l+1
        int wc = network->getWeightsOffset(l+1) - network->getWeightsOffset(l);
        double *wdiff = weightDiffs + network->getWeightsOffset(l);
        for (int i = 0; i<wc; i++) {
            wdiff[i] = -learningRate * nextLocalGradient[i%nextNeurons] * thisInput[i/nextNeurons];
        }
        
        // COMPUTE BIAS DERIVATIVES for layer l+1
        if (useBias) {
            for (int i = 0; i<nextNeurons; i++) {
                biasDiff[nextInputIdx + i] = -learningRate * nextLocalGradient[i];
            }
        }
        
        // COMPUTE LOCAL GRADIENTS for layer l
        
        // compute derivatives of neuron inputs in layer l
        double *thisInputDerivatives = new double[thisNeurons];
        daf(thisInput, thisInputDerivatives, thisNeurons);
        
        // compute local gradients for layer l
        for (int i = 0; i<thisNeurons; i++) {
            double sumNextGradient = 0;
            for (int j = 0; j<nextNeurons; j++) {
                sumNextGradient += nextLocalGradient[j] * weights[i * thisNeurons + j];
            }
            thisLocalGradient[i] = sumNextGradient * thisInputDerivatives[i];
        }
    }
}

void BackpropagationLearner::adjustWeights() {
    int wc = network->getWeightsOffset(network->getConfiguration()->getLayers());
    double *weights = network->getWeights();
    LOG()->debug("Adjusting weights by: [[%f, %f], [%f, %f]], [[%f, %f]].",
            weightDiffs[2], weightDiffs[3],
            weightDiffs[4], weightDiffs[5],
            weightDiffs[6], weightDiffs[7]);
    
    // we should skip the garbage in zero-layer weights
    for(int i = network->getWeightsOffset(1); i<wc; i++) {
        weights[i] += weightDiffs[i];
    }
}

void BackpropagationLearner::adjustBias() {
    double *bias = network->getBiasValues();
    int noNeurons = network->getAllNeurons();
    for (int i = 0; i<noNeurons; i++) {
        bias[i] += biasDiff[i];
    }
}

void BackpropagationLearner::clearLayer(double *inputPtr, int layerSize) {
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

void BackpropagationLearner::setTargetMse(double mse) {
    targetMse = mse;
}
