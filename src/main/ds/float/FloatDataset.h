/* 
 * File:   FloatDataset.h
 * Author: janvojt
 *
 * Created on October 11, 2016, 9:14 PM
 */

#ifndef FLOATDATASET_H
#define	FLOATDATASET_H

#include "../LabeledDataset.h"

#include <iostream>
#include <fstream>

using namespace std;

/**
 * Dataset containing set of float values, each representing a single neuron
 * potential. Each float value is stored in binary, hence the dataset file
 * is binary. This dataset does not load all data into memory at once, instead
 * only reads the patterns being currently processed. This makes it suitable
 * for working with big data.
 * 
 * The first 4 bytes represent an integer determining the size of input.
 * The next 4 bytes represent an integer determining the size of output.
 * This header is followed by the data without any delimiters.
 * 
 * The dataset implementation does not try to resolve big/little endians
 * in any way. It is recommended to use the dataset on the same endianness
 * it was created on.
 * 
 * @param filePath path to the dataset file stored on disk
 */
class FloatDataset : public LabeledDataset {
public:
    FloatDataset(char* filePath);
    FloatDataset(const FloatDataset& orig);
    virtual ~FloatDataset();
    
    /**
     * Clones the dataset by creating a shallow copy (point to the same data).
     * 
     * @return shallow copy
     */
    FloatDataset* clone();
    
    /**
     * @return Size of a single input pattern.
     */
    int getInputDimension();
    
    /**
     * @return Size of a single pattern label.
     */
    int getOutputDimension();

    /**
     * @return the next pattern followed by its label
     */
    data_t *next();
    
    /**
     * @return whether there is next pattern
     */
    bool hasNext();
    
    /**
     * Resets the cursor to the beginning of the array.
     */
    void reset();

private:
    int inDimension = 0;
    int outDimension = 0;
    int width = 0;
    int patternBytes = 0;
    ifstream fp;
};

#endif	/* FLOATDATASET_H */

