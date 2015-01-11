/* 
 * File:   IdxData.cpp
 * Author: janvojt
 * 
 * Created on January 7, 2015, 12:17 AM
 */

#include "IdxData.h"

IdxData::IdxData() {
}

IdxData::IdxData(const IdxData& orig) {
}

IdxData::~IdxData() {
    if (dimensionSizes != NULL) delete[] dimensionSizes;
    if (data != NULL) delete[] data;
}

int IdxData::getNoDimensions() {
    return this->noDimensions;
}

void IdxData::setNoDimensions(int noDimensions) {
    this->noDimensions = noDimensions;
    if (this->dimensionSizes != NULL) {
        delete[] dimensionSizes;
    }
    this->dimensionSizes = new int[noDimensions];
}

int IdxData::getDimensionSize(int dimension) {
    return this->dimensionSizes[dimension];
}

void IdxData::setDimensionSize(int dimension, int dimensionSize) {
    this->dimensionSizes[dimension] = dimensionSize;
}

char IdxData::getDataType() {
    return this->dataType;
}

void IdxData::setDataType(char dataType) {
    this->dataType = dataType;
}

void* IdxData::getData() {
    return this->data;
}

void IdxData::initData() {
    int size = dimensionSizes[0];
    for (int i = 1; i<noDimensions; i++) {
        size *= dimensionSizes[i];
    }
    this->data = new char[size];
}

int IdxData::getDataSize() {
    int size = dimensionSizes[0];
    for (int i = 1; i<noDimensions; i++) {
        size *= dimensionSizes[i];
    }
    return size;
}
