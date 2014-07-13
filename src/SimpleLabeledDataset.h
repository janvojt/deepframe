/* 
 * File:   SimpleLabeledDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 8:17 PM
 */

#ifndef SIMPLELABELEDDATASET_H
#define	SIMPLELABELEDDATASET_H

#include "LabeledDataset.h"


class SimpleLabeledDataset : LabeledDataset {
public:
    SimpleLabeledDataset(int inputDimension, int outputDimension, int size);
    SimpleLabeledDataset(const SimpleLabeledDataset& orig);
    virtual ~SimpleLabeledDataset();
    int getInputDimension();
    int getOutputDimension();
    float *next();
    bool hasNext();
    void reset();
    void addPattern(float *input, float *output);
private:
    void initDataset();
    int inDimension;
    int outDimension;
    int cursor;
    int addedCounter;
    int size;
    float *data;

};

#endif	/* SIMPLELABELEDDATASET_H */

