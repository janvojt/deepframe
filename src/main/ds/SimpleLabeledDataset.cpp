/* 
 * File:   SimpleLabeledDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 8:17 PM
 */

#include "SimpleLabeledDataset.h"
#include <cstring>
#include <stdlib.h>

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
SimpleLabeledDataset<dType>::SimpleLabeledDataset() {

}

template <typename dType>
SimpleLabeledDataset<dType>::SimpleLabeledDataset(int inputDimension, int outputDimension, int size) {
    addedCounter = 0;
    inDimension = inputDimension;
    outDimension = outputDimension;
    this->size = size;
    data = new dType[(inputDimension + outputDimension) * size];
}

template <typename dType>
SimpleLabeledDataset<dType>::SimpleLabeledDataset(const SimpleLabeledDataset& orig) {
}

template <typename dType>
SimpleLabeledDataset<dType>::~SimpleLabeledDataset() {
    delete[] data;
}

template <typename dType>
void SimpleLabeledDataset<dType>::addPattern(const dType *input, const dType *output) {
    
    if (addedCounter >= size) {
        LOG()->error("Trying to add %d learning patterns while the dataset size is only %d.", addedCounter+1, size);
    }
    
    int inputSize = sizeof(dType) * inDimension;
    int outputSize = sizeof(dType) * outDimension;
    int patternSize = inDimension + outDimension;
    dType *dataPtr = data + (addedCounter * patternSize);
    std::memcpy(dataPtr, input, inputSize);
    std::memcpy(dataPtr + inDimension, output, outputSize);
    addedCounter++;
}

template <typename dType>
int SimpleLabeledDataset<dType>::getInputDimension() {
    return inDimension;
}

template <typename dType>
int SimpleLabeledDataset<dType>::getOutputDimension() {
    return outDimension;
}

template <typename dType>
void SimpleLabeledDataset<dType>::initDataset() {
    data = new dType[size * (inDimension + outDimension)];
}

template <typename dType>
dType* SimpleLabeledDataset<dType>::next() {
    return data + (cursor++ * (inDimension + outDimension));
}

template <typename dType>
bool SimpleLabeledDataset<dType>::hasNext() {
    return cursor < size;
}

template <typename dType>
void SimpleLabeledDataset<dType>::reset() {
    cursor = 0;
}

template <typename dType>
int SimpleLabeledDataset<dType>::getSize() {
    return size;
}


template <typename dType>
SimpleLabeledDataset<dType>* SimpleLabeledDataset<dType>::takeAway(int size) {
    
    // we cannot take more than we have
    if (size > this->size) {
        size = this->size;
    }
    
    // fix the size of this dataset
    this->size -= size;
    if (this->addedCounter > this->size) {
        this->addedCounter = this->size;
    }
    
    // create the new dataset
    SimpleLabeledDataset<dType> *ds = new SimpleLabeledDataset<dType>();
    ds->inDimension = this->inDimension;
    ds->outDimension = this->outDimension;
    ds->size = size;
    ds->data = this->data + (this->inDimension + this->outDimension) * this->size;
    
    return ds;
}

template <typename dType>
void SimpleLabeledDataset<dType>::shuffle() {
    
    int entrySize = inDimension + outDimension;
    int memSize = sizeof(dType) * entrySize;
    dType *swap = new dType[entrySize];
    
    int limit = size * entrySize;
    for (int i = 0; i<limit; i+=entrySize) {
        int rnd = ((dType) (rand()) / (RAND_MAX-1) * size);
        rnd *= entrySize;

        std::memcpy(swap, data+i, memSize);
        std::memcpy(data+i, data+rnd, memSize);
        std::memcpy(data+rnd, swap, memSize);
    }
}

INSTANTIATE_DATA_CLASS(SimpleLabeledDataset);