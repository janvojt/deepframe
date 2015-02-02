/* 
 * File:   FoldTrainingDataset.cpp
 * Author: janvojt
 * 
 * Created on February 2, 2015, 12:36 AM
 */

#include "FoldTrainingDataset.h"
#include "../InputDataset.h"
#include "../LabeledDataset.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FoldTrainingDataset::FoldTrainingDataset(LabeledDataset **folds, int k) {
    noFolds = k;
    this->folds = folds;
    valIdx = k-1;
    foldIdx = 0;
}

FoldTrainingDataset::FoldTrainingDataset(const FoldTrainingDataset& orig) {
}

FoldTrainingDataset::~FoldTrainingDataset() {
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


double* FoldTrainingDataset::next() {
    if (hasNext()) {
        return folds[foldIdx]->next();
    } else {
        return NULL;
    }
}

void FoldTrainingDataset::reset() {
    valIdx = nextFold(valIdx);
    foldIdx = nextFold(valIdx);
}

void FoldTrainingDataset::shuffle() {
    LOG()->error("Shuffling folded dataset is not supported. Please shuffle before folding.");
}

LabeledDataset* FoldTrainingDataset::takeAway(int size) {
    LOG()->error("Taking away from folded dataset is not supported. Please take away before folding.");
    return NULL;
}

int FoldTrainingDataset::nextFold(int currFold) {
    int nFoldIdx = currFold + 1;
    if (nFoldIdx >= noFolds) {
        nFoldIdx = 0;
    }
    return nFoldIdx;
}
