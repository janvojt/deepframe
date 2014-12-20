/* 
 * File:   SimpleLabeledDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 8:17 PM
 */

#ifndef SIMPLELABELEDDATASET_H
#define	SIMPLELABELEDDATASET_H

#include "LabeledDataset.h"

// Class for simple construction of small dataset
// with labeled data for supervised learning.
// To be used with small datasets in case we know the dataset size ahead.
class SimpleLabeledDataset : LabeledDataset {
public:
    // Constructor correctly initializes the dataset
    // according to input and output label dimensions and dataset size.
    SimpleLabeledDataset(int inputDimension, int outputDimension, int size);
    SimpleLabeledDataset(const SimpleLabeledDataset& orig);
    virtual ~SimpleLabeledDataset();
    int getInputDimension();
    int getOutputDimension();
    double *next();
    bool hasNext();
    void reset();
    // Adds a single input vector with its label.
    void addPattern(const double *input, const double *output);
private:
    void initDataset();
    int inDimension;
    int outDimension;
    int cursor;
    int addedCounter;
    int size;
    double *data;

};

#endif	/* SIMPLELABELEDDATASET_H */
