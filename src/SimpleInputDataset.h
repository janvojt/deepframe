/* 
 * File:   SimpleInputDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 8:07 PM
 */

#ifndef SIMPLEINPUTDATASET_H
#define	SIMPLEINPUTDATASET_H

#include "InputDataset.h"


class SimpleInputDataset : InputDataset {
public:
    SimpleInputDataset(int dimension, int size);
    SimpleInputDataset(const SimpleInputDataset& orig);
    virtual ~SimpleInputDataset();
    float *next();
    bool hasNext();
    void reset();
    int getInputDimension();
    void addInput(float *input);
private:
    void initDataset();
    int dimension;
    int cursor;
    int addedCounter;
    int size;
    float *data;
};

#endif	/* SIMPLEINPUTDATASET_H */

