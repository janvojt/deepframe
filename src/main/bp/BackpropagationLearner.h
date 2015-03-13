/* 
 * File:   BackpropagationLearner.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 12:10 AM
 */

#ifndef BACKPROPAGATIONLEARNER_H
#define	BACKPROPAGATIONLEARNER_H

#include "../net/Network.h"
#include "../ds/LabeledDataset.h"
#include "../err/ErrorComputer.h"

template <typename dType>
class BackpropagationLearner {
public:
    BackpropagationLearner(Network<dType> *network);
    BackpropagationLearner(const BackpropagationLearner& orig);
    virtual ~BackpropagationLearner();
    
    /**
     * Launches the learning process with given training set and validation set.
     * 
     * @param trainingSet dataset with patterns to be used for training
     * @param validationSet validation dataset used for calculating error
     * @param valIdx index of the fold used for validation,
     *  use zero if not using k-fold cross validation
     * @return error in the validation dataset
     *  (or training dataset if no validation dataset is given)
     */
    dType train(LabeledDataset<dType> *trainingSet, LabeledDataset<dType> *validationSet, int valIdx);
    // Sets the learning rate influencing speed and quality of learning.
    void setLearningRate(dType learningRate);
    // Sets the maximum number of epochs.
    void setEpochLimit(long limit);
    // Sets object for computing network error.
    void setErrorComputer(ErrorComputer<dType> *errorComputer);
    // Set target Mean Square Error. When it is reached, training is finished.
    void setTargetMse(dType mse);
    // Set the number of epochs after which error improvement is required
    // to continue the learning process.
    void setImproveEpochs(int improvementEpochs);
    // Set minimal improvement of MSE required to keep learning.
    void setDeltaError(dType deltaError);
protected:
    // We cannot reuse the forward run from the network's implementation,
    // because additional meta results need to be kept for backpropagation
    // algorithm.
    void doForwardPhase(dType *input);
    // Backward phase optimizing network parameters in the learning process.
    void doBackwardPhase(dType *expectedOutput);
    // Computes local gradients for output neurons.
    virtual void computeOutputGradients(dType *expectedOutput) = 0;
    // Computes total differential for all weights
    // and local gradients for hidden neurons.
    virtual void computeWeightDifferentials() = 0;
    // Adjust network weights according to computed total differentials.
    virtual void adjustWeights() = 0;
    // Adjust network bias according to computed total differentials.
    virtual void adjustBias() = 0;
    // Allocates memory for caching variables.
    virtual void allocateCache() = 0;
    // ANN itself. Used for accessing configuration and tuning weights.
    Network<dType> *network;
    // Learning parameter. Intended to be decreasing during learning process.
    dType learningRate;
    // Stop learning when given number of epochs passes.
    long epochLimit;
    // Target Mean Square Error. When it is reached, training is finished.
    dType targetMse;
    // Number of epochs after which at least deltaError improvement
    // is required to continue learning.
    int improveEpochs;
    // Minimal improvement of MSE required to keep learning.
    dType deltaError;
    // Total differential for weight adjustment.
    dType *weightDiffs;
    // Cache for local gradients of respective neurons.
    dType *localGradients;
    // Total differential for bias adjustment.
    dType *biasDiff;
    // Whether to use bias.
    bool useBias;
    // Computes the Mean Square Error for the output produced in network.
    ErrorComputer<dType> *errorComputer;
    // Number of layers in network.
    int noLayers;
private:
    // Validates input dataset provided for learning against neural network.
    void validate(LabeledDataset<dType> *dataset);
    // Checks whether there was sufficient error improvement during last epochs.
    bool isErrorImprovement(dType error, int epoch);
    
    /** Computes error on given dataset.
    
        @param ds Dataset with labels to run through the neural network
        and calculate error on.
    
        @return error computed by #errorComputer.
     */
    dType computeError(LabeledDataset<dType> *ds);
    
    // Cache with last error rates.
    dType *errorCache;
    // Cursor for iterating the error cache.
    int errorCachePtr;
};

#endif	/* BACKPROPAGATIONLEARNER_H */

