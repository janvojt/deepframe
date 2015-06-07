/* 
 * File:   BackpropagationLearner.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 12:10 AM
 */

#ifndef BACKPROPAGATIONLEARNER_H
#define	BACKPROPAGATIONLEARNER_H

#include "../common.h"
#include "../net/Network.h"
#include "../ds/LabeledDataset.h"
#include "../err/ErrorComputer.h"
#include "TrainingResult.h"

class BackpropagationLearner {
public:
    BackpropagationLearner(Network *network);
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
    TrainingResult* train(LabeledDataset *trainingSet, LabeledDataset *validationSet, int valIdx);
    // Sets the learning rate influencing speed and quality of learning.
    void setLearningRate(data_t learningRate);
    // Sets the maximum number of epochs.
    void setEpochLimit(long limit);
    // Sets object for computing network error.
    void setErrorComputer(ErrorComputer *errorComputer);
    // Set target Mean Square Error. When it is reached, training is finished.
    void setTargetMse(data_t mse);
    // Set the number of epochs after which error improvement is required
    // to continue the learning process.
    void setImproveEpochs(int improvementEpochs);
    // Set minimal improvement of MSE required to keep learning.
    void setDeltaError(data_t deltaError);
protected:
    // We cannot reuse the forward run from the network's implementation,
    // because additional meta results need to be kept for backpropagation
    // algorithm.
    void doForwardPhase(data_t *input);
    // Backward phase optimizing network parameters in the learning process.
    void doBackwardPhase(data_t *expectedOutput);
    // ANN itself. Used for accessing configuration and tuning weights.
    Network *network;
    // Learning parameter. Intended to be decreasing during learning process.
    data_t learningRate;
    // Stop learning when given number of epochs passes.
    long epochLimit;
    // Target Mean Square Error. When it is reached, training is finished.
    data_t targetMse;
    // Number of epochs after which at least deltaError improvement
    // is required to continue learning.
    int improveEpochs;
    // Minimal improvement of MSE required to keep learning.
    data_t deltaError;
    // Total differential for weight adjustment.
    data_t *weightDiffs;
    // Cache for local gradients of respective neurons.
    data_t *localGradients;
    // Total differential for bias adjustment.
    data_t *biasDiff;
    // Whether to use bias.
    bool useBias;
    // Computes the Mean Square Error for the output produced in network.
    ErrorComputer *errorComputer;
    // Number of layers in network.
    int noLayers;
private:
    // Validates input dataset provided for learning against neural network.
    void validate(LabeledDataset *dataset);
    // Checks whether there was sufficient error improvement during last epochs.
    bool isErrorImprovement(data_t error, int epoch);
    
    /** Computes error on given dataset.
    
        @param ds Dataset with labels to run through the neural network
        and calculate error on.
    
        @return error computed by #errorComputer.
     */
    data_t computeError(LabeledDataset *ds);
    
    // Cache with last error rates.
    data_t *errorCache;
    // Cursor for iterating the error cache.
    int errorCachePtr;
};

#endif	/* BACKPROPAGATIONLEARNER_H */

