/* 
 * File:   SimpleInputDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 8:07 PM
 */

#ifndef SIMPLEINPUTDATASET_H
#define	SIMPLEINPUTDATASET_H

#include "InputDataset.h"
#include "../common.h"

/**
 * Class used for construction of small datasets
 * with input data to be processed by the network.
 * 
 * @param dimension the size of a single pattern
 * @param size the number of patterns in the dataset
 */
class SimpleInputDataset : InputDataset {
public:
    SimpleInputDataset(int dimension, int size);
    SimpleInputDataset(const SimpleInputDataset& orig);
    virtual ~SimpleInputDataset();
    
    /**
     * @return the next pattern from the dataset and increments the cursor
     */
    data_t *next();
    
    /**
     * @return whether there is next pattern
     */
    bool hasNext();
    
    /**
     * Resets the cursor to the beginning of the dataset.
     */
    void reset();
    
    /**
     * @return the size of a single input pattern
     */
    int getInputDimension();
    
    /**
     * Adds a single input pattern into the dataset.
     * 
     * @param input input pattern
     */
    void addInput(const data_t *input);
    
private:
    void initDataset();
    int dimension;
    int cursor;
    int addedCounter;
    int size;
    data_t *data;
};

#endif	/* SIMPLEINPUTDATASET_H */

