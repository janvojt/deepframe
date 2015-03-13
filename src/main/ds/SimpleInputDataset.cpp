/* 
 * File:   SimpleInputDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 8:07 PM
 */

#include "SimpleInputDataset.h"

#include <cstring>

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
SimpleInputDataset<dType>::SimpleInputDataset(int dimension, int size) {
    this->dimension = dimension;
    this->size = size;
    this->cursor = 0;
    this->addedCounter = 0;
    this->initDataset();
}

template <typename dType>
SimpleInputDataset<dType>::SimpleInputDataset(const SimpleInputDataset& orig) {
}

template <typename dType>
SimpleInputDataset<dType>::~SimpleInputDataset() {
    delete[] data;
}

template <typename dType>
void SimpleInputDataset<dType>::initDataset() {
    data = new dType[dimension * size];
}

template <typename dType>
void SimpleInputDataset<dType>::addInput(const dType* input) {
    
    if (addedCounter >= size) {
        LOG()->error("Trying to add %d input patterns while the dataset size is only %d.", addedCounter+1, size);
    }
    
    int patternSize = sizeof(dType) * dimension;
    dType *dataPtr = data + (addedCounter * dimension);
    std::memcpy(dataPtr, input, patternSize);
    addedCounter++;
}

template <typename dType>
dType* SimpleInputDataset<dType>::next() {
    return data + (dimension * cursor++);
}

template <typename dType>
bool SimpleInputDataset<dType>::hasNext() {
    return cursor < size;
}

template <typename dType>
void SimpleInputDataset<dType>::reset() {
    cursor = 0;
}

template <typename dType>
int SimpleInputDataset<dType>::getInputDimension() {
    return dimension;
}

INSTANTIATE_DATA_CLASS(SimpleInputDataset);