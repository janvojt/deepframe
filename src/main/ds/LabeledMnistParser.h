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
    LabeledDataset *parse(char *filePath);
private:
};

#endif	/* LABELEDIDXPARSER_H */

