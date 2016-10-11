/* 
 * File:   InMemoryLabeledDataset.h
 * Author: janvojt
 *
 * Created on October 11, 2016, 9:29 PM
 */

#ifndef INMEMORYLABELEDDATASET_H
#define	INMEMORYLABELEDDATASET_H

#include "LabeledDataset.h"

class InMemoryLabeledDataset : public LabeledDataset {
public:
    InMemoryLabeledDataset();
    InMemoryLabeledDataset(const InMemoryLabeledDataset& orig);
    virtual ~InMemoryLabeledDataset();
    
    /**
     * Clones the dataset by creating a shallow copy (point to the same data).
     */
    virtual InMemoryLabeledDataset* clone() = 0;
    
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
    virtual InMemoryLabeledDataset *takeAway(int size) = 0;
    
    /** Shuffles all elements in the dataset so that they are ordered randomly.
     */
    virtual void shuffle() = 0;
private:

};

#endif	/* INMEMORYLABELEDDATASET_H */

