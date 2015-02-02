/* 
 * File:   FoldDatasetFactory.cpp
 * Author: janvojt
 * 
 * Created on February 2, 2015, 12:21 AM
 */

#include "FoldDatasetFactory.h"
#include "FoldTrainingDataset.h"
#include "FoldValidationDataset.h"

FoldDatasetFactory::FoldDatasetFactory(LabeledDataset *ds, int k) {
    noFolds = k;
    folds = new LabeledDataset*[k];
    int foldSize = ds->getSize() / k;
    for (int i = k-1; i>=0; i--) {
        folds[i] = ds->takeAway(foldSize);
    }
}

FoldDatasetFactory::FoldDatasetFactory(const FoldDatasetFactory& orig) {
}

FoldDatasetFactory::~FoldDatasetFactory() {
    delete[] folds;
}

FoldTrainingDataset* FoldDatasetFactory::getTrainingDataset() {
    return new FoldTrainingDataset(folds, noFolds);
}

FoldValidationDataset* FoldDatasetFactory::getValidationDataset() {
    return new FoldValidationDataset(folds, noFolds);
}
