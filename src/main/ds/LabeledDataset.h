/* 
 * File:   LabeledDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 5:47 PM
 */

#ifndef LABELEDDATASET_H
#define	LABELEDDATASET_H

#include "InputDataset.h"

// Represents the dataset with labeled input patterns
// to be used for supervised learning.
class LabeledDataset : public InputDataset {
public:
    LabeledDataset();
    LabeledDataset(const LabeledDataset& orig);
    virtual ~LabeledDataset();
    
    /**
     * Clones the dataset by creating a shallow copy (point to the same data).
     */
    virtual LabeledDataset* clone() = 0;
    
    /**
     * @return the dimension of labels for input data.
     */
    virtual int getOutputDimension() = 0;
private:

};

#endif	/* LABELEDDATASET_H */

