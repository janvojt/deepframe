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

SimpleLabeledDataset::SimpleLabeledDataset() {

}

SimpleLabeledDataset::SimpleLabeledDataset(int inputDimension, int outputDimension, int size) {
    addedCounter = 0;
    inDimension = inputDimension;
    outDimension = outputDimension;
    this->size = size;
    data = new data_t[(inputDimension + outputDimension) * size];
}

SimpleLabeledDataset::SimpleLabeledDataset(const SimpleLabeledDataset& orig) {
    
    this->inDimension = orig.inDimension;
    this->outDimension = orig.outDimension;
    this->size = orig.size;
    this->addedCounter = orig.addedCounter;
    this->data = data;
}

SimpleLabeledDataset* SimpleLabeledDataset::clone() {
    return new SimpleLabeledDataset(*this);
}

SimpleLabeledDataset::~SimpleLabeledDataset() {
    delete[] data;
}

void SimpleLabeledDataset::addPattern(const data_t *input, const data_t *output) {
    
    if (addedCounter >= size) {
        LOG()->error("Trying to add %d learning patterns while the dataset size is only %d.", addedCounter+1, size);
    }
    
    int inputSize = sizeof(data_t) * inDimension;
    int outputSize = sizeof(data_t) * outDimension;
    int patternSize = inDimension + outDimension;
    data_t *dataPtr = data + (addedCounter * patternSize);
    std::memcpy(dataPtr, input, inputSize);
    std::memcpy(dataPtr + inDimension, output, outputSize);
    addedCounter++;
}

int SimpleLabeledDataset::getInputDimension() {
    return inDimension;
}

int SimpleLabeledDataset::getOutputDimension() {
    return outDimension;
}

void SimpleLabeledDataset::initDataset() {
    data = new data_t[size * (inDimension + outDimension)];
}

data_t* SimpleLabeledDataset::next() {
    return data + (cursor++ * (inDimension + outDimension));
}

bool SimpleLabeledDataset::hasNext() {
    return cursor < size;
}

void SimpleLabeledDataset::reset() {
    cursor = 0;
}

int SimpleLabeledDataset::getSize() {
    return size;
}


SimpleLabeledDataset* SimpleLabeledDataset::takeAway(int size) {
    
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
    SimpleLabeledDataset *ds = new SimpleLabeledDataset();
    ds->inDimension = this->inDimension;
    ds->outDimension = this->outDimension;
    ds->size = size;
    ds->data = this->data + (this->inDimension + this->outDimension) * this->size;
    
    return ds;
}

void SimpleLabeledDataset::shuffle() {
    
    int entrySize = inDimension + outDimension;
    int memSize = sizeof(data_t) * entrySize;
    data_t *swap = new data_t[entrySize];
    
    int limit = size * entrySize;
    for (int i = 0; i<limit; i+=entrySize) {
        int rnd = ((data_t) (rand()) / (RAND_MAX-1) * size);
        rnd *= entrySize;

        std::memcpy(swap, data+i, memSize);
        std::memcpy(data+i, data+rnd, memSize);
        std::memcpy(data+rnd, swap, memSize);
    }
}
