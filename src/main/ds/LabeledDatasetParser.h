/* 
 * File:   LabeledDatasetParser.h
 * Author: janvojt
 *
 * Created on October 13, 2014, 9:53 PM
 */

#ifndef LABELEDDATASETPARSER_H
#define	LABELEDDATASETPARSER_H

#include "../net/NetworkConfiguration.h"
#include "LabeledDataset.h"


class LabeledDatasetParser {
public:
    LabeledDatasetParser(char *filepath, NetworkConfiguration *netConf);
    LabeledDatasetParser(const LabeledDatasetParser& orig);
    virtual ~LabeledDatasetParser();
    LabeledDataset *parse();
private:
    NetworkConfiguration *netConf;
    char *filepath;
};

#endif	/* LABELEDDATASETPARSER_H */

