/* 
 * File:   FoldValidationDataset.cpp
 * Author: janvojt
 * 
 * Created on February 1, 2015, 10:26 PM
 */

#include <assert.h>

#include "FoldValidationDataset.h"

#include "../../common.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FoldValidationDataset::FoldValidationDataset(LabeledDataset **folds, int k, int valIdx) {
    assert(valIdx >= 0 && valIdx < k);
    noFolds = k;
    this->folds = folds;
    this->valIdx = valIdx;
}

FoldValidationDataset::FoldValidationDataset(const FoldValidationDataset& orig) {
    this->noFolds = orig.noFolds;
    this->folds = orig.folds;
    this->valIdx = orig.valIdx;
}

FoldValidationDataset::~FoldValidationDataset() {
}

LabeledDataset* FoldValidationDataset::clone() {
    return new FoldValidationDataset(*this);
}

int FoldValidationDataset::getInputDimension() {
    folds[0]->getInputDimension();
}

int FoldValidationDataset::getOutputDimension() {
    folds[0]->getOutputDimension();
}

int FoldValidationDataset::getSize() {
    return noFolds * folds[0]->getSize();
}

bool FoldValidationDataset::hasNext() {
    return folds[valIdx]->hasNext();
}

data_t* FoldValidationDataset::next() {
    return folds[valIdx]->next();
}

void FoldValidationDataset::reset() {
    folds[valIdx]->reset();
}

void FoldValidationDataset::shuffle() {
    LOG()->error("Shuffling folded dataset is not supported. Please shuffle before folding.");
}

LabeledDataset* FoldValidationDataset::takeAway(int size) {
    LOG()->error("Taking away from folded dataset is not supported. Please take away before folding.");
    return NULL;
}
