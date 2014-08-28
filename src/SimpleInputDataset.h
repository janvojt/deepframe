/* 
 * File:   SimpleInputDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 8:07 PM
 */

#ifndef SIMPLEINPUTDATASET_H
#define	SIMPLEINPUTDATASET_H

#include "InputDataset.h"

// Class used for construction of small datasets
// with input data to be processed by the network.
class SimpleInputDataset : InputDataset {
public:
    SimpleInputDataset(int dimension, int size);
    SimpleInputDataset(const SimpleInputDataset& orig);
    virtual ~SimpleInputDataset();
    double *next();
    bool hasNext();
    void reset();
    int getInputDimension();
    // Adds a single input pattern to be processed by the network.
    void addInput(double *input);
private:
    void initDataset();
    int dimension;
    int cursor;
    int addedCounter;
    int size;
    double *data;
};

#endif	/* SIMPLEINPUTDATASET_H */

