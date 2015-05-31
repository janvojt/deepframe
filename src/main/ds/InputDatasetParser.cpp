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

InputDatasetParser::InputDatasetParser(char* filepath, NetworkConfiguration* netConf) {
    this->netConf = netConf;
    this->filepath = filepath;
}

InputDatasetParser::InputDatasetParser(const InputDatasetParser& orig) {
}

InputDatasetParser::~InputDatasetParser() {
}

InputDataset* InputDatasetParser::parse() {
    
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
    
    SimpleInputDataset *ds = new SimpleInputDataset(inNeurons, size);
    
    for (int l = 0; l<size; l++) {
        
        // read input
        data_t *input = new data_t[inNeurons];
        for (int i = 0; i<inNeurons; i++) {
            fp >> input[i]; // read number
        }
        
        // truncate LF or CRLF
        fp.get() == 13 && fp.get();
        
        ds->addInput((const data_t *) input);
        delete[] input;
    }
    
    fp.close();

    return (InputDataset *) ds;
}
