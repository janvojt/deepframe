/* 
 * File:   TrainigResult.cpp
 * Author: janvojt
 * 
 * Created on March 14, 2015, 6:51 PM
 */

#include "TrainingResult.h"

#include "../common.h"

TrainingResult::TrainingResult() {
}

TrainingResult::TrainingResult(const TrainingResult& orig) {
    this->epochs = orig.epochs;
    this->trainingError = orig.trainingError;
    this->validationError = orig.validationError;
}

TrainingResult::~TrainingResult() {
}

long TrainingResult::getEpochs() {
    return epochs;
}

void TrainingResult::setEpochs(long epochs) {
    this->epochs = epochs;
}

data_t TrainingResult::getTrainingError() {
    return trainingError;
}

void TrainingResult::setTrainingError(data_t error) {
    this->trainingError = error;
}

data_t TrainingResult::getValidationError() {
    return validationError;
}

void TrainingResult::setValidationError(data_t error) {
    this->validationError = error;
}

data_t TrainingResult::getError() {
    return validationError >= 0 ? validationError : trainingError;
}
