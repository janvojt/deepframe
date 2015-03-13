/* 
 * File:   LabeledDatasetParser.h
 * Author: janvojt
 *
 * Created on October 13, 2014, 9:53 PM
 */

#ifndef LABELEDDATASETPARSER_H
#define	LABELEDDATASETPARSER_H

#include "LabeledDataset.h"

#include "../net/NetworkConfiguration.h"

template <typename dType>
class LabeledDatasetParser {
public:
    LabeledDatasetParser(char *filepath, NetworkConfiguration<dType> *netConf);
    LabeledDatasetParser(const LabeledDatasetParser& orig);
    virtual ~LabeledDatasetParser();
    LabeledDataset<dType> *parse();
private:
    NetworkConfiguration<dType> *netConf;
    char *filepath;
};

#endif	/* LABELEDDATASETPARSER_H */

