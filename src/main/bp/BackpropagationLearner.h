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

class BackpropagationLearner {
public:
    BackpropagationLearner(Network *network);
    BackpropagationLearner(const BackpropagationLearner& orig);
    virtual ~BackpropagationLearner();
    // Launches the learning process with given training set and validation set.
    void train(LabeledDataset *trainingSet, LabeledDataset *validationSet);
    // Sets the learning rate influencing speed and quality of learning.
    void setLearningRate(double learningRate);
    // Sets the maximum number of epochs.
    void setEpochLimit(long limit);
    // Sets object for computing network error.
    void setErrorComputer(ErrorComputer *errorComputer);
    // Set target Mean Square Error. When it is reached, training is finished.
    void setTargetMse(double mse);
    // Set the number of epochs after which error improvement is required
    // to continue the learning process.
    void setImproveEpochs(int improvementEpochs);
    // Set minimal improvement of MSE required to keep learning.
    void setDeltaError(double deltaError);
protected:
    // We cannot reuse the forward run from the network's implementation,
    // because additional meta results need to be kept for backpropagation
    // algorithm.
    void doForwardPhase(double *input);
    // Backward phase optimizing network parameters in the learning process.
    void doBackwardPhase(double *expectedOutput);
    // Computes local gradients for output neurons.
    virtual void computeOutputGradients(double *expectedOutput) = 0;
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
    Network *network;
    // Learning parameter. Intended to be decreasing during learning process.
    double learningRate;
    // Represents average error of the current network configuration.
    double errorTotal;
    // Counter of epochs, incremented right before new epoch is started.
    long epochCounter;
    // Stop learning when given number of epochs passes.
    long epochLimit;
    // Target Mean Square Error. When it is reached, training is finished.
    double targetMse;
    // Number of epochs after which at least deltaError improvement
    // is required to continue learning.
    int improveEpochs;
    // Minimal improvement of MSE required to keep learning.
    double deltaError;
    // Total differential for weight adjustment.
    double *weightDiffs;
    // Cache for local gradients of respective neurons.
    double *localGradients;
    // Total differential for bias adjustment.
    double *biasDiff;
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
    bool isErrorImprovement(double error, int epoch);
    
    /** Computes error on given dataset.
    
        @param ds Dataset with labels to run through the neural network
        and calculate error on.
    
        @return error computed by #errorComputer.
     */
    double computeError(LabeledDataset *ds);
    
    // Cache with last error rates.
    double *errorCache;
    // Cursor for iterating the error cache.
    int errorCachePtr;
};

#endif	/* BACKPROPAGATIONLEARNER_H */

