/* 
 * File:   BackpropagationLearner.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 12:10 AM
 */

#ifndef BACKPROPAGATIONLEARNER_H
#define	BACKPROPAGATIONLEARNER_H

#include "Network.h"
#include "ds/LabeledDataset.h"
#include "ErrorComputer.h"

class BackpropagationLearner {
public:
    BackpropagationLearner(Network *network);
    BackpropagationLearner(const BackpropagationLearner& orig);
    virtual ~BackpropagationLearner();
    // Launches the learning process.
    void train(LabeledDataset *dataset);
    // Sets the maximum number of epochs.
    void setEpochLimit(long limit);
    // Sets object for computing network error.
    void setErrorComputer(ErrorComputer *errorComputer);
    // Set target Mean Square Error. When it is reached, training is finished.
    void setTargetMse(double mse);
private:
    // Validates input dataset provided for learning against neural network.
    void validate(LabeledDataset *dataset);
    // We cannot reuse the forward run from the network's implementation,
    // because additional meta results need to be kept for backpropagation
    // algorithm.
    void doForwardPhase(double *input);
    // Backward phase optimizing network parameters in the learning process.
    void doBackwardPhase(double *expectedOutput);
    // Computes local gradients for output neurons.
    void computeOutputGradients(double *expectedOutput);
    // Computes total differential for all weights
    // and local gradients for hidden neurons.
    void computeWeightDifferentials();
    // Adjust network weights according to computed total differentials.
    void adjustWeights();
    // Adjust network bias according to computed total differentials.
    void adjustBias();
    // Helper method for clearing network layer.
    void clearLayer(double *inputPtr, int layerSize);
    // Allocates memory for caching variables.
    void allocateCache();
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
};

#endif	/* BACKPROPAGATIONLEARNER_H */

