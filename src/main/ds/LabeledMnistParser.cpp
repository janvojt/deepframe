/* 
 * File:   LabeledIdxParser.cpp
 * Author: janvojt
 * 
 * Created on January 6, 2015, 8:32 PM
 */

#include "LabeledMnistParser.h"
#include "SimpleLabeledDataset.h"
#include "idx/IdxParser.h"

#include <iostream>
#include <fstream>
#include <string.h>
#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

using namespace std;

// Number of label classes (0-9 numerals for MNIST).
const unsigned int LABEL_CLASSES = 10;

// data dimensions
const unsigned int INPUT_DIMENSIONS = 3;
const unsigned int LABEL_DIMENSIONS = 1;

template <typename dType>
LabeledMnistParser<dType>::LabeledMnistParser() {
}

template <typename dType>
LabeledMnistParser<dType>::LabeledMnistParser(const LabeledMnistParser& orig) {
}

template <typename dType>
LabeledMnistParser<dType>::~LabeledMnistParser() {
}

template <typename dType>
LabeledDataset<dType>* LabeledMnistParser<dType>::parse(char *filePath) {
    
    IdxParser *parser = new IdxParser();
    
    // split the file paths
    char *dataPath, *labelPath;
    dataPath = strtok(filePath, ":");
    labelPath = strtok(NULL, ":");
    
    LOG()->info("Parsing IDX file '%s' to build the dataset with input patterns.", dataPath);
    IdxData *data = parser->parse(dataPath);
    if (data == NULL) {
        return (LabeledDataset<dType> *) new SimpleLabeledDataset<dType>(0, 0, 0);
    }
    
    LOG()->info("Parsing IDX file '%s' to build the dataset with labels.", labelPath);
    IdxData *labels = parser->parse(labelPath);
    if (labels == NULL) {
        return (LabeledDataset<dType> *) new SimpleLabeledDataset<dType>(0, 0, 0);
    }
    
    // validate input IDX data
    if (data->getNoDimensions() != INPUT_DIMENSIONS) {
        LOG()->error("IDX file with the input patterns has unexpected number of dimensions (actual %d, expected %d).", data->getNoDimensions(), INPUT_DIMENSIONS);
        return (LabeledDataset<dType> *) new SimpleLabeledDataset<dType>(0, 0, 0);
    }
    if (labels->getNoDimensions() != LABEL_DIMENSIONS) {
        LOG()->error("IDX file with the labels has unexpected number of dimensions (actual %d, expected %d).", labels->getNoDimensions(), LABEL_DIMENSIONS);
        return (LabeledDataset<dType> *) new SimpleLabeledDataset<dType>(0, 0, 0);
    }
    if (data->getDimensionSize(0) < labels->getDimensionSize(0)) {
        LOG()->error("IDX files with input patterns and labels have inconsistent dataset sizes (%d vs. %d).", data->getDimensionSize(0), labels->getDimensionSize(0));
        return (LabeledDataset<dType> *) new SimpleLabeledDataset<dType>(0, 0, 0);
    }
    
    // create the dataset consumable by feed-forward network
    int dims = data->getNoDimensions();
    int size = labels->getDimensionSize(0);
    int xdim = data->getDimensionSize(1);
    int ydim = data->getDimensionSize(2);
    int patternSize = xdim*ydim;
    
    SimpleLabeledDataset<dType> *ds = new SimpleLabeledDataset<dType>(patternSize, LABEL_CLASSES, size);
    unsigned char *pData = (unsigned char *) data->getData();
    unsigned char *pLabels = (unsigned char *) labels->getData();
    for (int i = 0; i<size; i++) {
        
        // prepare input image
        dType *in = new dType[patternSize];
        for (int j = 0; j<patternSize; j++) {
            // normalize gray pixel ranges from 0-255 to 0-1
            in[j] = ((dType) *pData) / 255;
            pData++;
        }
        
        // prepare output array with labels
        dType *out = new dType[LABEL_CLASSES];
        fill_n(out, LABEL_CLASSES, 0);
        if (*pLabels < LABEL_CLASSES) {
            out[*pLabels] = 1.0;
            pLabels++;
        }
        
        // assign to dataset
        ds->addPattern(in, out);
        
        // free memory
        delete in;
        delete out;
    }
    
    delete parser;
    
    return (LabeledDataset<dType> *) ds;
}

INSTANTIATE_DATA_CLASS(LabeledMnistParser);