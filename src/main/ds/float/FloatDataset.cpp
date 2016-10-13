/* 
 * File:   FloatDataset.cpp
 * Author: janvojt
 * 
 * Created on October 11, 2016, 9:14 PM
 */

#include "FloatDataset.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FloatDataset::FloatDataset(char* filePath) {

    // open IDX file with the dataset
    fp.open(filePath, ios::in|ios::binary);
    
    if (fp.is_open()) {
        
        // parse dimension sizes
        int *dims = new int[2];
        fp.read((char *) dims, 2 * sizeof(int));
        
        inDimension = dims[0];
        outDimension = dims[1];
        width = inDimension + outDimension;
        patternBytes = width * sizeof(data_t);
        
        LOG()->debug("Pattern input vs. output size read from dataset: %d x %d.", inDimension, outDimension);

        delete[] dims;
    } else {
        LOG()->error("Cannot open file '%s' for parsing.", filePath);
    }
}

FloatDataset::FloatDataset(const FloatDataset& orig) {
    this->inDimension = orig.inDimension;
    this->outDimension = orig.outDimension;
}

FloatDataset::~FloatDataset() {
    fp.close();
}

FloatDataset* FloatDataset::clone() {
    return new FloatDataset(*this);
}

int FloatDataset::getInputDimension() {
    return inDimension;
}

int FloatDataset::getOutputDimension() {
    return outDimension;
}

data_t* FloatDataset::next() {
    
    if (fp.eof()) {
        return NULL;
    }

    data_t *data = new data_t[width];
    fp.read((char *)data, patternBytes);

    return data;
}

bool FloatDataset::hasNext() {

    int c = fp.peek();
    if (c == EOF) {
        if (fp.eof()) {
            // end of file
            return false;
        } else {
            // error
            LOG()->error("Error reading from dataset.");
            return false;
        }
    }

    return true;
}

void FloatDataset::reset() {
    fp.clear();
    fp.seekg(2*sizeof(data_t), ios::beg);
}
