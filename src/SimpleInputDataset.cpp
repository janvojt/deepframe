/* 
 * File:   SimpleInputDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 8:07 PM
 */

#include "SimpleInputDataset.h"
#include <cstring>
#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

SimpleInputDataset::SimpleInputDataset(int dimension, int size) {
    this->dimension = dimension;
    this->size = size;
    this->cursor = 0;
    this->addedCounter = 0;
}

SimpleInputDataset::SimpleInputDataset(const SimpleInputDataset& orig) {
}

SimpleInputDataset::~SimpleInputDataset() {
}

void SimpleInputDataset::initDataset() {
    data = new double[dimension * size];
}

void SimpleInputDataset::addInput(double* input) {
    
    if (addedCounter >= size) {
        LOG()->error("Trying to add %d input patterns while the dataset size is only %d.", addedCounter+1, size);
    }
    
    int patternSize = sizeof(double) * dimension;
    double *dataPtr = data + (size * patternSize);
    std::memcpy(dataPtr, input, patternSize);
    addedCounter++;
}

double* SimpleInputDataset::next() {
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

