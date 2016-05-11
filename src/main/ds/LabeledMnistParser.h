/* 
 * File:   LabeledIdxParser.h
 * Author: janvojt
 *
 * Created on January 6, 2015, 8:32 PM
 */

#ifndef LABELEDIDXPARSER_H
#define	LABELEDIDXPARSER_H

#include "LabeledDataset.h"

class LabeledMnistParser {
public:
    LabeledMnistParser();
    LabeledMnistParser(const LabeledMnistParser& orig);
    virtual ~LabeledMnistParser();
    
    /**
     * Parses the MNIST dataset from file and stores it in memory.
     * 
     * @param filePath
     * @return the dataset in memory
     */
    LabeledDataset *parse(char *filePath);
    
private:
};

#endif	/* LABELEDIDXPARSER_H */

