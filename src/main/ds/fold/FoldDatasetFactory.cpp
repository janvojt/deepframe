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
    folds = new LabeledDataset<dType>*[k];
    int foldSize = ds->getSize() / k;
    for (int i = k-1; i>=0; i--) {
        folds[i] = ds->takeAway(foldSize);
    }
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