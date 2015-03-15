/* 
 * File:   FoldValidationDataset.h
 * Author: janvojt
 *
 * Created on February 1, 2015, 10:26 PM
 */

#ifndef FOLDVALIDATIONDATASET_H
#define	FOLDVALIDATIONDATASET_H

#include "../LabeledDataset.h"

/** Validation dataset for k-fold cross validation. */
template <typename dType>
class FoldValidationDataset : LabeledDataset<dType> {
public:
    
    /** Constructor building the validation dataset from given folds.
        
        @param folds array of labeled datasets representing respective folds
        @param k number of folds
        @param valIdx specifies which fold is fixed for validation
     */
    FoldValidationDataset(LabeledDataset<dType> **folds, int k, int valIdx);
    
    /** Copy constructor.
        
        @param orig original validation dataset to clone from
     */
    FoldValidationDataset(const FoldValidationDataset& orig);
    
    /** Destructor releasing memory. */
    virtual ~FoldValidationDataset();
    
    /**
     * @return shallow copy
     */
    virtual LabeledDataset<dType>* clone();

    
    /** Gets the input pattern dimension.
        
        @return input dimensions
     */
    int getInputDimension();
    
    /** Gets the output label dimension.
        
        @return output label dimension
     */
    int getOutputDimension();
    
    /** Gets the next pattern to be process according to the order specified
        by the k-fold cross validation algorithm.
        
        @return next pattern with respective label
     */
    dType *next();
    
    /** Determines whether there are still unprocessed patterns in the
        validation dataset.
        
        @return true if there is next pattern from the validation dataset,
        false otherwise.
     */
    bool hasNext();
    
    /** Moves the validation dataset pointer to the next fold. */
    void reset();
    
    /** Gets the total number of all patterns in all folds.
        
        @return size of all patterns in the dataset
     */
    int getSize();
    
    /** Unsupported operation. */
    void shuffle();
    
    /** Unsupported operation. */
    LabeledDataset<dType> *takeAway(int size);
    
private:
    
    /** Number of folds in the dataset. */
    int noFolds;
    
    /** Array with dataset folds. */
    LabeledDataset<dType> **folds;
    
    /** Pointer to the current validation dataset fold. */
    int valIdx;
};

#endif	/* FOLDVALIDATIONDATASET_H */

