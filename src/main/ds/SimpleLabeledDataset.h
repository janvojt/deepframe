/* 
 * File:   SimpleLabeledDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 8:17 PM
 */

#ifndef SIMPLELABELEDDATASET_H
#define	SIMPLELABELEDDATASET_H

#include "LabeledDataset.h"
#include "../common.h"

// Class for simple construction of small dataset
// with labeled data for supervised learning.
// To be used with small datasets in case we know the dataset size ahead.
class SimpleLabeledDataset : LabeledDataset {
public:
    // Constructor correctly initializes the dataset
    // according to input and output label dimensions and dataset size.
    SimpleLabeledDataset(int inputDimension, int outputDimension, int size);
    SimpleLabeledDataset(const SimpleLabeledDataset& orig);
    
    /**
     * Clones the dataset by creating a shallow copy (point to the same data).
     * 
     * @return shallow copy
     */
    SimpleLabeledDataset* clone();
    
    virtual ~SimpleLabeledDataset();
    int getInputDimension();
    int getOutputDimension();
    data_t *next();
    bool hasNext();
    void reset();
    int getSize();
    // Adds a single input vector with its label.
    void addPattern(const data_t *input, const data_t *output);
    
    /** Creates a new dataset taking patterns from this dataset.

        This method is useful for creating validation sets.
        
        @param size number of patterns taken from the end of this dataset
        that is put into the newly created dataset
        
        @return the newly created dataset
     */
    SimpleLabeledDataset* takeAway(int size);
    
    /** Shuffles all elements in the dataset so that they are ordered randomly.
     */
    void shuffle();
    
private:
    SimpleLabeledDataset();
    void initDataset();
    int inDimension;
    int outDimension;
    int cursor;
    int addedCounter;
    int size;
    data_t *data;

};

#endif	/* SIMPLELABELEDDATASET_H */

