/* 
 * File:   FoldDatasetFactory.cpp
 * Author: janvojt
 * 
 * Created on February 2, 2015, 12:21 AM
 */

#include "FoldDatasetFactory.h"

#include "FoldTrainingDataset.h"
#include "FoldValidationDataset.h"
#include "../../common.h"

FoldDatasetFactory::FoldDatasetFactory(InMemoryLabeledDataset *ds, int k) {
    
    noFolds = k;
    
    InMemoryLabeledDataset *copy = ds->clone();
    
    folds = new InMemoryLabeledDataset*[k];
    int foldSize = copy->getSize() / k;
    for (int i = k-1; i>=0; i--) {
        folds[i] = copy->takeAway(foldSize);
    }
    
    delete copy;
}

FoldDatasetFactory::FoldDatasetFactory(const FoldDatasetFactory& orig) {
}

FoldDatasetFactory::~FoldDatasetFactory() {
    delete[] folds;
}

FoldTrainingDataset* FoldDatasetFactory::getTrainingDataset(int valIdx) {
    return new FoldTrainingDataset(folds, noFolds, valIdx);
}

FoldValidationDataset* FoldDatasetFactory::getValidationDataset(int valIdx) {
    return new FoldValidationDataset(folds, noFolds, valIdx);
}
