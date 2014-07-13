/* 
 * File:   InputDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 5:29 PM
 */

#include "InputDataset.h"
#include <cstring>

InputDataset::InputDataset(int dimension) {
    this->dimension = dimension;
    this->size = size;
    this->cursor = 0;
}

InputDataset::InputDataset(const InputDataset& orig) {
}

InputDataset::~InputDataset() {
    delete data;
}

void InputDataset::initDataset() {
    data = new float[dimension * size];
}

void InputDataset::addInput(float* input) {
    std::memcpy(data, input, sizeof(float) * dimension);
    size++;
}

float* InputDataset::next() {
    return data + (dimension * cursor++);
}

bool InputDataset::hasNext() {
    return cursor >= size;
}
