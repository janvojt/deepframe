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

using namespace std;

InputDatasetParser::InputDatasetParser(char* filepath, NetworkConfiguration* netConf) {
    this->netConf = netConf;
    this->filepath = filepath;
}

InputDatasetParser::InputDatasetParser(const InputDatasetParser& orig) {
}

InputDatasetParser::~InputDatasetParser() {
}

InputDataset* InputDatasetParser::parse() {
    
    ifstream fp(filepath);
    LOG()->info("Parsing file '%s' for test dataset.", filepath);

    if (!fp.is_open()) {
        LOG()->error("Unable to open file %s.", filepath);
        return NULL;
    }

    int inNeurons = 0;
    int size = 0;
    
    // read dataset size
    fp >> size;

    // truncate LF or CRLF
    fp.get() == 13 && fp.get();
        
    // read input size
    if (size > 0) {
        
        int beginData = fp.tellg();
        char ch;
        while ((ch = fp.peek()) != std::char_traits<char>::eof())
        {
            if (ch == '\n') {
                break;
            } else if (ch == ' '|| ch == '\t' || ch == '\r') {
                char truncate;
                fp.read(&truncate, 1);
            } else if (ch == '>') {
                LOG()->warn("Using data file with labels as a testing dataset is not currently supported. Parsing will most likely fail.");
                char truncate;
                fp.read(&truncate, 1);
                break;
            }
            else
            {
                data_t truncate;
                fp >> truncate;
                inNeurons++;
            }
        }
        fp.seekg(beginData);
    }
    
    LOG()->info("Reading data file %s with %d inputs.", filepath, inNeurons);
    
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
