/* 
 * File:   BackpropeagationLearner.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 12:10 AM
 */

#include "BackpropagationLearner.h"

#include <cstring>
#include <string>
#include <stdexcept>
#include <limits>

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

BackpropagationLearner::BackpropagationLearner(Network *network) {
    this->network = network;
    learningRate = 1;
    epochCounter = 0;
    epochLimit = 1000000;
    targetMse = .0001;
    errorTotal = std::numeric_limits<double>::infinity();
    useBias = network->getConfiguration()->getBias();
    noLayers = network->getConfiguration()->getLayers();
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    if (errorComputer != NULL) delete errorComputer;
}

void BackpropagationLearner::train(LabeledDataset *dataset) {
    double mse = 0;
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
        LOG()->debug("Starting forward phase for dataset %d in epoch %d.", datasetSize, epochCounter);
            doForwardPhase(pattern);
        LOG()->debug("Starting backward phase for dataset %d in epoch %d.", datasetSize, epochCounter);
            doBackwardPhase(expOutput);
            mse += errorComputer->compute(network, expOutput);
        }
        mse = mse / datasetSize;
        LOG()->info("Finished epoch %d with MSE: %f.", epochCounter, mse);
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

void BackpropagationLearner::validate(LabeledDataset *dataset) {
    if (dataset->getInputDimension() != network->getInputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same input dimension as the number of input neurons!");
    }
    if (dataset->getOutputDimension() != network->getOutputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same output dimension as the number of output neurons!");
    }
}

void BackpropagationLearner::setEpochLimit(long limit) {
    epochLimit = limit;
}

void BackpropagationLearner::setErrorComputer(ErrorComputer* errorComputer) {
    this->errorComputer = errorComputer;
}

void BackpropagationLearner::setTargetMse(double mse) {
    targetMse = mse;
}

void BackpropagationLearner::setLearningRate(double learningRate) {
    this->learningRate = learningRate;
}
