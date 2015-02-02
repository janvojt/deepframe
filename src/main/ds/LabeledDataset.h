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
    // Returns the dimension of labels for input data.
    virtual int getOutputDimension() = 0;
    
    /** Gets the dataset size.
        
        @return number of patterns in this dataset.
     */
    virtual int getSize() = 0;
    
    /** Creates a new dataset taking patterns from this dataset.

        This method is useful for creating validation sets.
        
        @param size
        number of patterns taken from the end of this dataset
        that is put into the newly created dataset
        
        @return the newly created dataset
     */
    virtual LabeledDataset *takeAway(int size) = 0;
    
    /** Shuffles all elements in the dataset so that they are ordered randomly.
     */
    virtual void shuffle() = 0;
private:

};

#endif	/* LABELEDDATASET_H */

