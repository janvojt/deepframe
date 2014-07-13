/* 
 * File:   LabeledDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 5:47 PM
 */

#ifndef LABELEDDATASET_H
#define	LABELEDDATASET_H

#include "InputDataset.h"


class LabeledDataset : public InputDataset {
public:
    LabeledDataset(int dimension);
    LabeledDataset(const LabeledDataset& orig);
    virtual ~LabeledDataset();
private:

};

#endif	/* LABELEDDATASET_H */

