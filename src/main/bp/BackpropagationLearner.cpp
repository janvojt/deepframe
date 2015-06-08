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

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

const int MAX_IMPROVEMENT_EPOCHS = 1000;

BackpropagationLearner::BackpropagationLearner(Network *network) {
    this->network = network;
    this->learningRate = 1;
    this->epochLimit = 1000000;
    this->targetMse = .0001;
    this->useBias = network->getConfiguration()->getBias();
    this->noLayers = network->getConfiguration()->getLayers();
    this->deltaError = 0;
    this->improveEpochs = 0;
}

BackpropagationLearner::BackpropagationLearner(const BackpropagationLearner &orig) {
}

BackpropagationLearner::~BackpropagationLearner() {
    if (this->errorComputer != NULL) delete this->errorComputer;
    if (this->improveEpochs > 0) delete[] this->errorCache;
}

TrainingResult* BackpropagationLearner::train(LabeledDataset *trainingSet, LabeledDataset *validationSet, int valIdx) {
    
    TrainingResult *result = new TrainingResult();
    if (epochLimit <= 0) {
        LOG()->info("Learning skipped based on zero epoch limit.");
        return result;
    } 
    
    LOG()->info("Started training with:\n"
            "   - cross-validation fold: %d,\n"
            "   - epoch limit: %d,\n"
            "   - target MSE: %f,\n"
            "   - epochs in which improvement is required: %d,\n"
            "   - learning rate: %f."
            , valIdx, this->epochLimit, this->targetMse, this->improveEpochs, this->learningRate);
    
    long epochCounter = 0;
    do {
        
        epochCounter++;
        LOG()->debug("Validation fold %d: Starting epoch %d.", valIdx, epochCounter);
        
        trainingSet->reset();
        int datasetSize = 0;
        data_t mse = 0;
        
        while (trainingSet->hasNext()) {
            datasetSize++;
            data_t *pattern = trainingSet->next();
            data_t *expOutput = pattern + trainingSet->getInputDimension();
            
            LOG()->debug("Validation fold %d: Starting forward phase for dataset %d in epoch %d.", valIdx, datasetSize, epochCounter);
            doForwardPhase(pattern);
            
            LOG()->debug("Validation fold %d: Starting backward phase for dataset %d in epoch %d.", valIdx, datasetSize, epochCounter);
            doBackwardPhase(expOutput);
            
            mse += this->errorComputer->compute(this->network, expOutput);
        }
        
        // compute MSE on training data
        mse = mse / datasetSize;
        result->setTrainingError(mse);
        LOG()->info("Validation fold %d: Finished epoch %d with MSE: %f.", valIdx, epochCounter, result->getTrainingError());
        
        // calculate error on validation dataset
        validationSet->reset();
        if (validationSet->hasNext()) {
            result->setValidationError(computeError(validationSet));
            LOG()->info("Validation fold %d: Computed MSE of %f on validation dataset.", valIdx, result->getValidationError());
        }
    
        // check criteria for stopping learning
        if (result->getError() <= this->targetMse) {
            LOG()->info("Validation fold %d: Training successful after %d epochs with MSE of %f.", valIdx, epochCounter, result->getError());
            break;
        } else if (!isErrorImprovement(result->getError(), epochCounter)) {
            LOG()->info("Validation fold %d: Training interrupted after %d epochs with MSE of %f, because MSE improvement in last %d epochs was less than %f.", valIdx, epochCounter, result->getError(), improveEpochs, deltaError);
            break;
        } else if (epochCounter >= this->epochLimit) {
            LOG()->info("Validation fold %d: Training interrupted after %d epochs with MSE of %f.", valIdx, epochCounter, result->getError());
            break;
        }
        
    } while (true); // stopping checks are at the end of the loop
    
    result->setEpochs(epochCounter);
    return result;
}

void BackpropagationLearner::doForwardPhase(data_t *input) {
    this->network->setInput(input);
    this->network->forward();
}

void BackpropagationLearner::doBackwardPhase(data_t *expectedOutput) {
    this->network->setExpectedOutput(expectedOutput);
    this->network->backward();
}

void BackpropagationLearner::validate(LabeledDataset *dataset) {
    if (dataset->getInputDimension() != this->network->getInputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same input dimension as the number of input neurons!");
    }
    if (dataset->getOutputDimension() != this->network->getOutputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same output dimension as the number of output neurons!");
    }
}

bool BackpropagationLearner::isErrorImprovement(data_t error, int epoch) {
    if (this->improveEpochs <= 0) {
        return true;
    }
    
    if (epoch > this->improveEpochs) {
        if ((this->errorCache[this->errorCachePtr] - this->deltaError) < error) {
            return false;
        } 
    }
    
    this->errorCache[this->errorCachePtr] = error;
    this->errorCachePtr = (this->errorCachePtr+1) % this->improveEpochs;
    
    return true;
}

data_t BackpropagationLearner::computeError(LabeledDataset* ds) {
    int datasetSize = 0;
    data_t vMse = 0;
    while (ds->hasNext()) {
        datasetSize++;
        data_t *pattern = ds->next();
        data_t *expOutput = pattern + ds->getInputDimension();
        vMse += this->errorComputer->compute(this->network, expOutput);
    }
    
    return vMse / datasetSize;
}

void BackpropagationLearner::setImproveEpochs(int improveEpochs) {
    if (improveEpochs > MAX_IMPROVEMENT_EPOCHS) {
        LOG()->warn("Allowed maximum for error improvement epochs is %d, however %d was  requested. Going with %d.", MAX_IMPROVEMENT_EPOCHS, improveEpochs, MAX_IMPROVEMENT_EPOCHS);
        improveEpochs = MAX_IMPROVEMENT_EPOCHS;
    }
    
    if (this->improveEpochs > 0) {
        delete[] this->errorCache;
    }
    
    if (improveEpochs > 0) {
        this->errorCache = new data_t[improveEpochs];
        this->errorCachePtr = 0;
    }
    
    this->improveEpochs = improveEpochs;
}

void BackpropagationLearner::setEpochLimit(long limit) {
    this->epochLimit = limit;
}

void BackpropagationLearner::setErrorComputer(ErrorComputer* errorComputer) {
    this->errorComputer = errorComputer;
}

void BackpropagationLearner::setTargetMse(data_t mse) {
    this->targetMse = mse;
}

void BackpropagationLearner::setLearningRate(data_t learningRate) {
    this->learningRate = learningRate;
}

void BackpropagationLearner::setDeltaError(data_t deltaError) {
    this->deltaError = deltaError;
}
