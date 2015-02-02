/* 
 * File:   FoldValidationDataset.cpp
 * Author: janvojt
 * 
 * Created on February 1, 2015, 10:26 PM
 */

#include "FoldValidationDataset.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FoldValidationDataset::FoldValidationDataset(LabeledDataset **folds, int k) {
    noFolds = k;
    this->folds = folds;
    valIdx = k-1;
}

FoldValidationDataset::FoldValidationDataset(const FoldValidationDataset& orig) {
}

FoldValidationDataset::~FoldValidationDataset() {
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

double* FoldValidationDataset::next() {
    return folds[valIdx]->next();
}

void FoldValidationDataset::reset() {
    valIdx++;
    if (valIdx >= noFolds) {
        valIdx = 0;
    }
    folds[valIdx]->reset();
}

void FoldValidationDataset::shuffle() {
    LOG()->error("Shuffling folded dataset is not supported. Please shuffle before folding.");
}

LabeledDataset* FoldValidationDataset::takeAway(int size) {
    LOG()->error("Taking away from folded dataset is not supported. Please take away before folding.");
    return NULL;
}
