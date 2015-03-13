/* 
 * File:   LabeledDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 5:47 PM
 */

#include "LabeledDataset.h"

#include "../common.h"

template <typename dType>
LabeledDataset<dType>::LabeledDataset() : InputDataset<dType>() {
}

template <typename dType>
LabeledDataset<dType>::LabeledDataset(const LabeledDataset& orig) : InputDataset<dType>(orig) {
}

template <typename dType>
LabeledDataset<dType>::~LabeledDataset() {
}

INSTANTIATE_DATA_CLASS(LabeledDataset);