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

FoldTrainingDataset::FoldTrainingDataset(LabeledDataset **folds, int k, int valIdx) {
    assert(valIdx >= 0 && valIdx < k);
    noFolds = k;
    this->folds = folds;
    this->valIdx = valIdx;
    foldIdx = 0;
}

FoldTrainingDataset::FoldTrainingDataset(const FoldTrainingDataset& orig) {
    this->noFolds = orig.noFolds;
    this->folds = orig.folds;
    this->valIdx = orig.valIdx;
    this->foldIdx = orig.foldIdx;
}

FoldTrainingDataset::~FoldTrainingDataset() {
}

FoldTrainingDataset* FoldTrainingDataset::clone() {
    return new FoldTrainingDataset(*this);
}

int FoldTrainingDataset::getInputDimension() {
    return folds[0]->getInputDimension();
}

int FoldTrainingDataset::getOutputDimension() {
    return folds[0]->getOutputDimension();
}

int FoldTrainingDataset::getSize() {
    return noFolds * folds[0]->getSize();
}

bool FoldTrainingDataset::hasNext() {
    if (folds[foldIdx]->hasNext()) {
        return true;
    }
    
    int nFoldIdx = nextFold(foldIdx);
    if (nFoldIdx != valIdx) {
        foldIdx = nFoldIdx;
    }
    
    return folds[foldIdx]->hasNext();
}

data_t* FoldTrainingDataset::next() {
    if (hasNext()) {
        return folds[foldIdx]->next();
    } else {
        return NULL;
    }
}

void FoldTrainingDataset::reset() {
    foldIdx = nextFold(valIdx);
    while (foldIdx != valIdx) {
        folds[foldIdx]->reset();
        foldIdx = nextFold(foldIdx);
    }
    foldIdx = nextFold(valIdx);
}

void FoldTrainingDataset::shuffle() {
    LOG()->error("Shuffling folded dataset is not supported. Please shuffle before folding.");
}

LabeledDataset* FoldTrainingDataset::takeAway(int size) {
    LOG()->error("Taking away from folded dataset is not supported. Please take away before folding.");
    return NULL;
}

/** Determines the next fold in line.
    
    @param refFold referential fold to move by 1 from
    @return next fold in line from refFold
 */
int FoldTrainingDataset::nextFold(int refFold) {
    int nFoldIdx = refFold + 1;
    if (nFoldIdx >= noFolds) {
        nFoldIdx = 0;
    }
    return nFoldIdx;
}
