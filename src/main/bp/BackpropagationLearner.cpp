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

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

const int MAX_IMPROVEMENT_EPOCHS = 1000;

BackpropagationLearner::BackpropagationLearner(Network *network) {
    this->network = network;
    learningRate = 1;
    epochLimit = 1000000;
    targetMse = .0001;
    useBias = network->getConfiguration()->getBias();
    noLayers = network->getConfiguration()->getLayers();
    deltaError = 0;
    improveEpochs = 0;
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    if (errorComputer != NULL) delete errorComputer;
    if (improveEpochs > 0) delete[] errorCache;
}

double BackpropagationLearner::train(LabeledDataset *trainingSet, LabeledDataset *validationSet, int valIdx) {
    
    long epochCounter = 0;
    double mse = 1.0; // maximum possible error
    LOG()->info("Started training with:\n"
            "   - cross-validation fold: %d,\n"
            "   - epoch limit: %d,\n"
            "   - target MSE: %f,\n"
            "   - epochs in which improvement is required: %d,\n"
            "   - learning rate: %f."
            , valIdx, epochLimit, targetMse, improveEpochs, learningRate);
    
    do {
        epochCounter++;
        LOG()->debug("Validation fold %d: Starting epoch %d.", valIdx, epochCounter);
        
        trainingSet->reset();
        int datasetSize = 0;
        mse = 0;
        while (trainingSet->hasNext()) {
            datasetSize++;
            double *pattern = trainingSet->next();
            double *expOutput = pattern + trainingSet->getInputDimension();
            
            LOG()->debug("Validation fold %d: Starting forward phase for dataset %d in epoch %d.", valIdx, datasetSize, epochCounter);
            doForwardPhase(pattern);
            
            LOG()->debug("Validation fold %d: Starting backward phase for dataset %d in epoch %d.", valIdx, datasetSize, epochCounter);
            doBackwardPhase(expOutput);
            
            mse += errorComputer->compute(network, expOutput);
        }
        mse = mse / datasetSize;
        LOG()->info("Validation fold %d: Finished epoch %d with MSE: %f.", valIdx, epochCounter, mse);
        
        // calculate error on validation dataset
        validationSet->reset();
        if (validationSet->hasNext()) {
            mse = computeError(validationSet);
            LOG()->info("Validation fold %d: Computed MSE of %f on validation dataset.", valIdx, mse);
        }
    
        // check criteria for stopping learning
        if (mse <= targetMse) {
            LOG()->info("Validation fold %d: Training successful after %d epochs with MSE of %f.", valIdx, epochCounter, mse);
            break;
        } else if (!isErrorImprovement(mse, epochCounter)) {
            LOG()->info("Validation fold %d: Training interrupted after %d epochs with MSE of %f, because MSE improvement in last %d epochs was less than %f.", valIdx, epochCounter, mse, improveEpochs, deltaError);
            break;
        } else if (epochCounter >= epochLimit) {
            LOG()->info("Validation fold %d: Training interrupted after %d epochs with MSE of %f.", valIdx, epochCounter, mse);
            break;
        }
        
    } while (true); // stopping checks are at the end of the loop
    
    return mse;
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

bool BackpropagationLearner::isErrorImprovement(double error, int epoch) {
    if (improveEpochs <= 0) {
        return true;
    }
    
    if (epoch > improveEpochs) {
        if ((errorCache[errorCachePtr] - deltaError) < error) {
            return false;
        } 
    }
    
    errorCache[errorCachePtr] = error;
    errorCachePtr = (errorCachePtr+1) % improveEpochs;
    
    return true;
}

double BackpropagationLearner::computeError(LabeledDataset* ds) {
    int datasetSize = 0;
    double vMse = 0;
    while (ds->hasNext()) {
        datasetSize++;
        double *pattern = ds->next();
        double *expOutput = pattern + ds->getInputDimension();
        vMse += errorComputer->compute(network, expOutput);
    }
    
    return vMse / datasetSize;
}

void BackpropagationLearner::setImproveEpochs(int improveEpochs) {
    if (improveEpochs > MAX_IMPROVEMENT_EPOCHS) {
        LOG()->warn("Allowed maximum for error improvement epochs is %d, however %d was  requested. Going with %d.", MAX_IMPROVEMENT_EPOCHS, improveEpochs, MAX_IMPROVEMENT_EPOCHS);
        improveEpochs = MAX_IMPROVEMENT_EPOCHS;
    }
    
    if (this->improveEpochs > 0) {
        delete[] errorCache;
    }
    
    if (improveEpochs > 0) {
        errorCache = new double[improveEpochs];
        errorCachePtr = 0;
    }
    
    this->improveEpochs = improveEpochs;
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

void BackpropagationLearner::setDeltaError(double deltaError) {
    this->deltaError = deltaError;
}
