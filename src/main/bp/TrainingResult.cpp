/* 
 * File:   TrainigResult.cpp
 * Author: janvojt
 * 
 * Created on March 14, 2015, 6:51 PM
 */

#include "TrainingResult.h"

#include "../common.h"

template <typename dType>
TrainingResult<dType>::TrainingResult() {
}

template <typename dType>
TrainingResult<dType>::TrainingResult(const TrainingResult& orig) {
    this->epochs = orig.epochs;
    this->trainingError = orig.trainingError;
    this->validationError = orig.validationError;
}

template <typename dType>
TrainingResult<dType>::~TrainingResult() {
}

template<typename dType>
long TrainingResult<dType>::getEpochs() {
    return epochs;
}

template<typename dType>
void TrainingResult<dType>::setEpochs(long epochs) {
    this->epochs = epochs;
}

template<typename dType>
dType TrainingResult<dType>::getTrainingError() {
    return trainingError;
}

template<typename dType>
void TrainingResult<dType>::setTrainingError(dType error) {
    this->trainingError = error;
}

template<typename dType>
dType TrainingResult<dType>::getValidationError() {
    return validationError;
}

template<typename dType>
void TrainingResult<dType>::setValidationError(dType error) {
    this->validationError = error;
}


template<typename dType>
dType TrainingResult<dType>::getError() {
    return validationError >= 0 ? validationError : trainingError;
}

INSTANTIATE_DATA_CLASS(TrainingResult);