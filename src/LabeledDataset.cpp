/* 
 * File:   LabeledDataset.cpp
 * Author: janvojt
 * 
 * Created on July 13, 2014, 5:47 PM
 */

#include "LabeledDataset.h"

LabeledDataset::LabeledDataset(int dimension) : InputDataset(dimension) {
}

LabeledDataset::LabeledDataset(const LabeledDataset& orig) : InputDataset(orig) {
}

LabeledDataset::~LabeledDataset() {
}
