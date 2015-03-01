/* 
 * File:   FoldTrainingDataset.h
 * Author: janvojt
 *
 * Created on February 2, 2015, 12:36 AM
 */

#ifndef FOLDTRAININGDATASET_H
#define	FOLDTRAININGDATASET_H

#include "../LabeledDataset.h"

/** Training dataset for k-fold cross validation. */
class FoldTrainingDataset : LabeledDataset {
    
public:
    
    /** Constructor building the training dataset from given folds.
        
        @param folds array of labeled datasets representing respective folds
        @param k number of folds
        @param valIdx specifies which fold is fixed for validation
     */
    FoldTrainingDataset(LabeledDataset **folds, int k, int valIdx);
    
    /** Copy constructor.
        
        @param orig original training dataset to clone from
     */
    FoldTrainingDataset(const FoldTrainingDataset& orig);
    
    /** Destructor releasing memory. */
    virtual ~FoldTrainingDataset();
    
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
    double *next();
    
    /** Determines whether there are still unprocessed patterns in the
        training dataset folds.
        
        @return true if there is next pattern from the training dataset,
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
    LabeledDataset *takeAway(int size);

private:

    /** Performs the roll-over to compute the next fold index.
        
        @param currFold current fold index
        @return the successive fold index
     */
    int nextFold(int currFold);
    
    /** Number of folds in the dataset. */
    int noFolds;
    
    /** Array with dataset folds. */
    LabeledDataset **folds;
    
    /** Pointer to the current training dataset fold. */
    int foldIdx;
    
    /** Pointer to the current validation dataset fold. */
    int valIdx;

};

#endif	/* FOLDTRAININGDATASET_H */

