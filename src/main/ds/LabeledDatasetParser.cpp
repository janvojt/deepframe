/* 
 * File:   LabeledDatasetParser.cpp
 * Author: janvojt
 * 
 * Created on October 13, 2014, 9:53 PM
 */

#include <iostream>
#include <fstream>

#include "LabeledDatasetParser.h"
#include "SimpleLabeledDataset.h"
#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

LabeledDatasetParser::LabeledDatasetParser(char* filepath, NetworkConfiguration* netConf) {
    this->netConf = netConf;
    this->filepath = filepath;
}

LabeledDatasetParser::LabeledDatasetParser(const LabeledDatasetParser& orig) {
}

LabeledDatasetParser::~LabeledDatasetParser() {
}

LabeledDataset* LabeledDatasetParser::parse() {
    
    std::ifstream fp(filepath);
    LOG()->info("Parsing file '%s' for labeled dataset.", filepath);

    if (!fp.is_open()) {
        LOG()->error("Unable to open file %s.", filepath);
        return NULL;
    }

    int inNeurons = netConf->getNeurons(0);
    int outNeurons = netConf->getNeurons(netConf->getLayers()-1);
    int size = 0;
    
    // read dataset size
    fp >> size;
    
    SimpleLabeledDataset *ds = new SimpleLabeledDataset(inNeurons, outNeurons, size);
    
    for (int l = 0; l<size; l++) {
        
        // read input
        double *input = new double[inNeurons];
        for (int i = 0; i<inNeurons; i++) {
            fp >> input[i]; // read number
        }
        
        // truncate separator
        fp.get(); // space
        fp.get(); // >
        fp.get(); // space

        // read output
        double *output = new double[outNeurons];
        for (int i = 0; i<outNeurons; i++) {
            fp >> output[i];
        }
        
        // truncate LF or CRLF
        fp.get() == 13 && fp.get();

        ds->addPattern(input, output);
    }
    
    fp.close();

    return (LabeledDataset *) ds;
}
