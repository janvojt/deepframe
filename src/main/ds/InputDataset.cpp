/* 
 * File:   InputDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 5:29 PM
 */

#include "InputDataset.h"

#include "../common.h"

template <typename dType>
InputDataset<dType>::InputDataset() {
}

template <typename dType>
InputDataset<dType>::InputDataset(const InputDataset& orig) {
}

template <typename dType>
InputDataset<dType>::~InputDataset() {
}

INSTANTIATE_DATA_CLASS(InputDataset);