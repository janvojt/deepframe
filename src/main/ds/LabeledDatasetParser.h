/* 
 * File:   LabeledDatasetParser.h
 * Author: janvojt
 *
 * Created on October 13, 2014, 9:53 PM
 */

#ifndef LABELEDDATASETPARSER_H
#define	LABELEDDATASETPARSER_H

#include "InMemoryLabeledDataset.h"

#include "../net/NetworkConfiguration.h"

class LabeledDatasetParser {
public:
    LabeledDatasetParser(char *filepath, NetworkConfiguration *netConf);
    LabeledDatasetParser(const LabeledDatasetParser& orig);
    virtual ~LabeledDatasetParser();
    
    /**
     * Parses the labeled dataset into memory.
     * 
     * @return the dataset in memory
     */
    InMemoryLabeledDataset *parse();
    
private:
    NetworkConfiguration *netConf;
    char *filepath;
};

#endif	/* LABELEDDATASETPARSER_H */

