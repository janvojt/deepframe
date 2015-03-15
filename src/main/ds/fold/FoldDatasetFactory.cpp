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

template <typename dType>
FoldDatasetFactory<dType>::FoldDatasetFactory(LabeledDataset<dType> *ds, int k) {
    
    noFolds = k;
    
    LabeledDataset<dType> *copy = ds->clone();
    
    folds = new LabeledDataset<dType>*[k];
    int foldSize = copy->getSize() / k;
    for (int i = k-1; i>=0; i--) {
        folds[i] = copy->takeAway(foldSize);
    }
    
    delete copy;
}

template <typename dType>
FoldDatasetFactory<dType>::FoldDatasetFactory(const FoldDatasetFactory& orig) {
}

template <typename dType>
FoldDatasetFactory<dType>::~FoldDatasetFactory() {
    delete[] folds;
}

template <typename dType>
FoldTrainingDataset<dType>* FoldDatasetFactory<dType>::getTrainingDataset(int valIdx) {
    return new FoldTrainingDataset<dType>(folds, noFolds, valIdx);
}

template <typename dType>
FoldValidationDataset<dType>* FoldDatasetFactory<dType>::getValidationDataset(int valIdx) {
    return new FoldValidationDataset<dType>(folds, noFolds, valIdx);
}

INSTANTIATE_DATA_CLASS(FoldDatasetFactory);