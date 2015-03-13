/* 
 * File:   InputDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 5:29 PM
 */

#ifndef INPUTDATASET_H
#define	INPUTDATASET_H

// Represents the dataset containing input patterns
// to be processed by neural network.
template <typename dType>
class InputDataset {
public:
    InputDataset();
    InputDataset(const InputDataset& orig);
    virtual ~InputDataset();
    // Returns a pointer to an array of next input values in the dataset.
    // Method is virtual so it is implementation agnostic.
    virtual dType *next() = 0;
    // Tells whether the dataset contains more input patterns to process.
    virtual bool hasNext() = 0;
    // Resets the cursor to the beginning of dataset.
    virtual void reset() = 0;
    // Returns the dimension of input pattern.
    virtual int getInputDimension() = 0;
private:
    
};

#endif	/* INPUTDATASET_H */

