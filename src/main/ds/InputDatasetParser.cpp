/* 
 * File:   InputDatasetParser.cpp
 * Author: janvojt
 * 
 * Created on October 14, 2014, 10:09 PM
 */

#include "InputDatasetParser.h"
#include "SimpleInputDataset.h"
#include "InputDataset.h"

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

#include <iostream>
#include <fstream>

template <typename dType>
InputDatasetParser<dType>::InputDatasetParser(char* filepath, NetworkConfiguration<dType>* netConf) {
    this->netConf = netConf;
    this->filepath = filepath;
}

template <typename dType>
InputDatasetParser<dType>::InputDatasetParser(const InputDatasetParser& orig) {
}

template <typename dType>
InputDatasetParser<dType>::~InputDatasetParser() {
}

template <typename dType>
InputDataset<dType>* InputDatasetParser<dType>::parse() {
    
    std::ifstream fp(filepath);
    LOG()->info("Parsing file '%s' for test dataset.", filepath);

    if (!fp.is_open()) {
        LOG()->error("Unable to open file %s.", filepath);
        return NULL;
    }

    int inNeurons = netConf->getNeurons(0);
    int size = 0;
    
    // read dataset size
    fp >> size;
    
    SimpleInputDataset<dType> *ds = new SimpleInputDataset<dType>(inNeurons, size);
    
    for (int l = 0; l<size; l++) {
        
        // read input
        dType *input = new dType[inNeurons];
        for (int i = 0; i<inNeurons; i++) {
            fp >> input[i]; // read number
        }
        
        // truncate LF or CRLF
        fp.get() == 13 && fp.get();
        
        ds->addInput((const dType *) input);
        delete[] input;
    }
    
    fp.close();

    return (InputDataset<dType> *) ds;
}

INSTANTIATE_DATA_CLASS(InputDatasetParser);