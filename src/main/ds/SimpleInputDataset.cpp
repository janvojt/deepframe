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

SimpleInputDataset::SimpleInputDataset(int dimension, int size) {
    this->dimension = dimension;
    this->size = size;
    this->cursor = 0;
    this->addedCounter = 0;
    this->initDataset();
}

SimpleInputDataset::SimpleInputDataset(const SimpleInputDataset& orig) {
}

SimpleInputDataset::~SimpleInputDataset() {
    delete[] data;
}

void SimpleInputDataset::initDataset() {
    data = new data_t[dimension * size];
}

void SimpleInputDataset::addInput(const data_t* input) {
    
    if (addedCounter >= size) {
        LOG()->error("Trying to add %d input patterns while the dataset size is only %d.", addedCounter+1, size);
    }
    
    int patternSize = sizeof(data_t) * dimension;
    data_t *dataPtr = data + (addedCounter * dimension);
    std::memcpy(dataPtr, input, patternSize);
    addedCounter++;
}

data_t* SimpleInputDataset::next() {
    return data + (dimension * cursor++);
}

bool SimpleInputDataset::hasNext() {
    return cursor < size;
}

void SimpleInputDataset::reset() {
    cursor = 0;
}

int SimpleInputDataset::getInputDimension() {
    return dimension;
}
