/* 
 * File:   FoldTrainingDataset.cpp
 * Author: janvojt
 * 
 * Created on February 2, 2015, 12:36 AM
 */

#include <assert.h>

#include "FoldTrainingDataset.h"
#include "../InputDataset.h"
#include "../LabeledDataset.h"

#include "../../common.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
FoldTrainingDataset<dType>::FoldTrainingDataset(LabeledDataset<dType> **folds, int k, int valIdx) {
    assert(valIdx >= 0 && valIdx < k);
    noFolds = k;
    this->folds = folds;
    this->valIdx = valIdx;
    foldIdx = 0;
}

template <typename dType>
FoldTrainingDataset<dType>::FoldTrainingDataset(const FoldTrainingDataset& orig) {
}

template <typename dType>
FoldTrainingDataset<dType>::~FoldTrainingDataset() {
}

template <typename dType>
int FoldTrainingDataset<dType>::getInputDimension() {
    return folds[0]->getInputDimension();
}

template <typename dType>
int FoldTrainingDataset<dType>::getOutputDimension() {
    return folds[0]->getOutputDimension();
}

template <typename dType>
int FoldTrainingDataset<dType>::getSize() {
    return noFolds * folds[0]->getSize();
}

template <typename dType>
bool FoldTrainingDataset<dType>::hasNext() {
    if (folds[foldIdx]->hasNext()) {
        return true;
    }
    
    int nFoldIdx = nextFold(foldIdx);
    if (nFoldIdx != valIdx) {
        foldIdx = nFoldIdx;
    }
    
    return folds[foldIdx]->hasNext();
}

template <typename dType>
dType* FoldTrainingDataset<dType>::next() {
    if (hasNext()) {
        return folds[foldIdx]->next();
    } else {
        return NULL;
    }
}

template <typename dType>
void FoldTrainingDataset<dType>::reset() {
    foldIdx = nextFold(valIdx);
    while (foldIdx != valIdx) {
        folds[foldIdx]->reset();
        foldIdx = nextFold(foldIdx);
    }
    foldIdx = nextFold(valIdx);
}

template <typename dType>
void FoldTrainingDataset<dType>::shuffle() {
    LOG()->error("Shuffling folded dataset is not supported. Please shuffle before folding.");
}

template <typename dType>
LabeledDataset<dType>* FoldTrainingDataset<dType>::takeAway(int size) {
    LOG()->error("Taking away from folded dataset is not supported. Please take away before folding.");
    return NULL;
}

/** Determines the next fold in line.
    
    @param refFold referential fold to move by 1 from
    @return next fold in line from refFold
 */
template <typename dType>
int FoldTrainingDataset<dType>::nextFold(int refFold) {
    int nFoldIdx = refFold + 1;
    if (nFoldIdx >= noFolds) {
        nFoldIdx = 0;
    }
    return nFoldIdx;
}

INSTANTIATE_DATA_CLASS(FoldTrainingDataset);