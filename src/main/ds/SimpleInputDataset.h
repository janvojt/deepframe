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

// Class used for construction of small datasets
// with input data to be processed by the network.
class SimpleInputDataset : InputDataset {
public:
    SimpleInputDataset(int dimension, int size);
    SimpleInputDataset(const SimpleInputDataset& orig);
    virtual ~SimpleInputDataset();
    data_t *next();
    bool hasNext();
    void reset();
    int getInputDimension();
    // Adds a single input pattern to be processed by the network.
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

