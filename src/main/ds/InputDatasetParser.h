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

template <typename dType>
class InputDatasetParser {
public:
    InputDatasetParser(char* filepath, NetworkConfiguration<dType>* netConf);
    InputDatasetParser(const InputDatasetParser& orig);
    virtual ~InputDatasetParser();
    InputDataset<dType> *parse();
private:
    NetworkConfiguration<dType> *netConf;
    char *filepath;
};

#endif	/* INPUTDATASETPARSER_H */

