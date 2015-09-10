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
#include "../common.h"

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

    int inNeurons = 0;
    int outNeurons = 0;
    int size = 0;
    
    // read dataset size
    fp >> size;

    // truncate LF or CRLF
    fp.get() == 13 && fp.get();
        
    // read input size
    if (size > 0) {
        
        int beginData = fp.tellg();
        bool countingInput = true;
        char ch;
        while ((ch = fp.peek()) != std::char_traits<char>::eof())
        {
            if (ch == '\n') {
                break;
            } else if (ch == ' '|| ch == '\t' || ch == '\r') {
                char truncate;
                fp.read(&truncate, 1);
            } else if (ch == '>') {
                char truncate;
                fp.read(&truncate, 1);
                countingInput = false;
            }
            else
            {
                data_t truncate;
                fp >> truncate;
                if (countingInput) {
                    inNeurons++;
                } else {
                    outNeurons++;
                }
            }
        }
        fp.seekg(beginData);
    }
    
    LOG()->info("Reading data file %s with %d inputs and %d outputs.", filepath, inNeurons, outNeurons);
    
    SimpleLabeledDataset *ds = new SimpleLabeledDataset(inNeurons, outNeurons, size);
    
    for (int l = 0; l<size; l++) {
        
        // read input
        data_t *input = new data_t[inNeurons];
        for (int i = 0; i<inNeurons; i++) {
            fp >> input[i]; // read number
        }
        
        // truncate separator
        fp.get(); // space
        fp.get(); // >
        fp.get(); // space

        // read output
        data_t *output = new data_t[outNeurons];
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
