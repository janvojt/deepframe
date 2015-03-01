/* 
 * File:   FoldDatasetFactory.h
 * Author: janvojt
 *
 * Created on February 2, 2015, 12:21 AM
 */

#ifndef FOLDDATASETFACTORY_H
#define	FOLDDATASETFACTORY_H

#include "../LabeledDataset.h"
#include "FoldTrainingDataset.h"
#include "FoldValidationDataset.h"

/** Factory for building k-fold training and validation datasets.
 
    The implementation of k-fold algorithm is in the datasets themselves.
    The folding occurs when reset() is called on the respective datasets.
    The correct functionality depends on calling the reset() only when
    it is needed!
 */
class FoldDatasetFactory {
public:
    
    /** Constructor preparing folds for the to-be-produced datasets.
        
        @param ds original labeled dataset (without folds)
        @param k number of folds
     */
    FoldDatasetFactory(LabeledDataset *ds, int k);
    
    /** Copy contructor.
     
        @param orig original factory to be cloned
     */
    FoldDatasetFactory(const FoldDatasetFactory& orig);
    
    /** Destructor releasing memory dedicated for folds. */
    virtual ~FoldDatasetFactory();
    
    /** Factory for the k-fold training dataset.
     
        @param valIdx specifies the fold index to use as validation dataset
        @return k-fold training dataset
     */
    FoldTrainingDataset *getTrainingDataset(int valIdx);
    
    /** Factory for the k-fold validation dataset.
        
        @param valIdx specifies the fold index to use as validation dataset
        @return k-fold validation dataset
     */
    FoldValidationDataset *getValidationDataset(int valIdx);
    
private:
    /** Number of folds in the dataset. */
    int noFolds;
    /** Array of dataset folds, over which validation is rotated. */
    LabeledDataset **folds;
};

#endif	/* FOLDDATASETFACTORY_H */

