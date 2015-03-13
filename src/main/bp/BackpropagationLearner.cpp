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

template <typename dType>
BackpropagationLearner<dType>::BackpropagationLearner(Network<dType> *network) {
    this->network = network;
    this->learningRate = 1;
    this->epochLimit = 1000000;
    this->targetMse = .0001;
    this->useBias = network->getConfiguration()->getBias();
    this->noLayers = network->getConfiguration()->getLayers();
    this->deltaError = 0;
    this->improveEpochs = 0;
}

template <typename dType>
BackpropagationLearner<dType>::BackpropagationLearner(const BackpropagationLearner &orig) {
}

template <typename dType>
BackpropagationLearner<dType>::~BackpropagationLearner() {
    if (this->errorComputer != NULL) delete this->errorComputer;
    if (this->improveEpochs > 0) delete[] this->errorCache;
}

template <typename dType>
dType BackpropagationLearner<dType>::train(LabeledDataset<dType> *trainingSet, LabeledDataset<dType> *validationSet, int valIdx) {
    
    long epochCounter = 0;
    dType mse = 1.0; // maximum possible error
    LOG()->info("Started training with:\n"
            "   - cross-validation fold: %d,\n"
            "   - epoch limit: %d,\n"
            "   - target MSE: %f,\n"
            "   - epochs in which improvement is required: %d,\n"
            "   - learning rate: %f."
            , valIdx, this->epochLimit, this->targetMse, this->improveEpochs, this->learningRate);
    
    do {
        epochCounter++;
        LOG()->debug("Validation fold %d: Starting epoch %d.", valIdx, epochCounter);
        
        trainingSet->reset();
        int datasetSize = 0;
        mse = 0;
        while (trainingSet->hasNext()) {
            datasetSize++;
            dType *pattern = trainingSet->next();
            dType *expOutput = pattern + trainingSet->getInputDimension();
            
            LOG()->debug("Validation fold %d: Starting forward phase for dataset %d in epoch %d.", valIdx, datasetSize, epochCounter);
            doForwardPhase(pattern);
            
            LOG()->debug("Validation fold %d: Starting backward phase for dataset %d in epoch %d.", valIdx, datasetSize, epochCounter);
            doBackwardPhase(expOutput);
            
            mse += this->errorComputer->compute(this->network, expOutput);
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
        if (mse <= this->targetMse) {
            LOG()->info("Validation fold %d: Training successful after %d epochs with MSE of %f.", valIdx, epochCounter, mse);
            break;
        } else if (!isErrorImprovement(mse, epochCounter)) {
            LOG()->info("Validation fold %d: Training interrupted after %d epochs with MSE of %f, because MSE improvement in last %d epochs was less than %f.", valIdx, epochCounter, mse, improveEpochs, deltaError);
            break;
        } else if (epochCounter >= this->epochLimit) {
            LOG()->info("Validation fold %d: Training interrupted after %d epochs with MSE of %f.", valIdx, epochCounter, mse);
            break;
        }
        
    } while (true); // stopping checks are at the end of the loop
    
    return mse;
}

template <typename dType>
void BackpropagationLearner<dType>::doForwardPhase(dType *input) {
    this->network->setInput(input);
    this->network->run();
}

template <typename dType>
void BackpropagationLearner<dType>::doBackwardPhase(dType *expectedOutput) {
    computeOutputGradients(expectedOutput);
    computeWeightDifferentials();
    adjustWeights();
    if (this->network->getConfiguration()->getBias()) {
        adjustBias();
    }
}

template <typename dType>
void BackpropagationLearner<dType>::validate(LabeledDataset<dType> *dataset) {
    if (dataset->getInputDimension() != this->network->getInputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same input dimension as the number of input neurons!");
    }
    if (dataset->getOutputDimension() != this->network->getOutputNeurons()) {
        throw new std::invalid_argument("Provided dataset must have the same output dimension as the number of output neurons!");
    }
}

template <typename dType>
bool BackpropagationLearner<dType>::isErrorImprovement(dType error, int epoch) {
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

template <typename dType>
dType BackpropagationLearner<dType>::computeError(LabeledDataset<dType>* ds) {
    int datasetSize = 0;
    dType vMse = 0;
    while (ds->hasNext()) {
        datasetSize++;
        dType *pattern = ds->next();
        dType *expOutput = pattern + ds->getInputDimension();
        vMse += this->errorComputer->compute(this->network, expOutput);
    }
    
    return vMse / datasetSize;
}

template <typename dType>
void BackpropagationLearner<dType>::setImproveEpochs(int improveEpochs) {
    if (improveEpochs > MAX_IMPROVEMENT_EPOCHS) {
        LOG()->warn("Allowed maximum for error improvement epochs is %d, however %d was  requested. Going with %d.", MAX_IMPROVEMENT_EPOCHS, improveEpochs, MAX_IMPROVEMENT_EPOCHS);
        improveEpochs = MAX_IMPROVEMENT_EPOCHS;
    }
    
    if (this->improveEpochs > 0) {
        delete[] this->errorCache;
    }
    
    if (improveEpochs > 0) {
        this->errorCache = new dType[improveEpochs];
        this->errorCachePtr = 0;
    }
    
    this->improveEpochs = improveEpochs;
}

template <typename dType>
void BackpropagationLearner<dType>::setEpochLimit(long limit) {
    this->epochLimit = limit;
}

template <typename dType>
void BackpropagationLearner<dType>::setErrorComputer(ErrorComputer<dType>* errorComputer) {
    this->errorComputer = errorComputer;
}

template <typename dType>
void BackpropagationLearner<dType>::setTargetMse(dType mse) {
    this->targetMse = mse;
}

template <typename dType>
void BackpropagationLearner<dType>::setLearningRate(dType learningRate) {
    this->learningRate = learningRate;
}

template <typename dType>
void BackpropagationLearner<dType>::setDeltaError(dType deltaError) {
    this->deltaError = deltaError;
}

INSTANTIATE_DATA_CLASS(BackpropagationLearner);