/* 
 * File:   InputDatasetParser.h
 * Author: janvojt
 *
 * Created on October 14, 2014, 10:09 PM
 */

#ifndef INPUTDATASETPARSER_H
#define	INPUTDATASETPARSER_H

#include "InputDataset.h"
#include "../net/NetworkConfiguration.h"

class InputDatasetParser {
public:
    InputDatasetParser(char* filepath, NetworkConfiguration* netConf);
    InputDatasetParser(const InputDatasetParser& orig);
    virtual ~InputDatasetParser();
    InputDataset *parse();
private:
    NetworkConfiguration *netConf;
    char *filepath;
};

#endif	/* INPUTDATASETPARSER_H */

