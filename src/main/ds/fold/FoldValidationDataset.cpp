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

template <typename dType>
FoldValidationDataset<dType>::FoldValidationDataset(LabeledDataset<dType> **folds, int k, int valIdx) {
    assert(valIdx >= 0 && valIdx < k);
    noFolds = k;
    this->folds = folds;
    this->valIdx = valIdx;
}

template <typename dType>
FoldValidationDataset<dType>::FoldValidationDataset(const FoldValidationDataset& orig) {
    this->noFolds = orig.noFolds;
    this->folds = orig.folds;
    this->valIdx = orig.valIdx;
}

template <typename dType>
FoldValidationDataset<dType>::~FoldValidationDataset() {
}

template<typename dType>
LabeledDataset<dType>* FoldValidationDataset<dType>::clone() {
    return new FoldValidationDataset<dType>(*this);
}

template <typename dType>
int FoldValidationDataset<dType>::getInputDimension() {
    folds[0]->getInputDimension();
}

template <typename dType>
int FoldValidationDataset<dType>::getOutputDimension() {
    folds[0]->getOutputDimension();
}

template <typename dType>
int FoldValidationDataset<dType>::getSize() {
    return noFolds * folds[0]->getSize();
}

template <typename dType>
bool FoldValidationDataset<dType>::hasNext() {
    return folds[valIdx]->hasNext();
}

template <typename dType>
dType* FoldValidationDataset<dType>::next() {
    return folds[valIdx]->next();
}

template <typename dType>
void FoldValidationDataset<dType>::reset() {
    folds[valIdx]->reset();
}

template <typename dType>
void FoldValidationDataset<dType>::shuffle() {
    LOG()->error("Shuffling folded dataset is not supported. Please shuffle before folding.");
}

template <typename dType>
LabeledDataset<dType>* FoldValidationDataset<dType>::takeAway(int size) {
    LOG()->error("Taking away from folded dataset is not supported. Please take away before folding.");
    return NULL;
}

INSTANTIATE_DATA_CLASS(FoldValidationDataset);