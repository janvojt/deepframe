/* 
 * File:   BackpropagationLearner.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 12:10 AM
 */

#ifndef BACKPROPAGATIONLEARNER_H
#define	BACKPROPAGATIONLEARNER_H

#include "Network.h"
#include "LabeledDataset.h"
#include <limits>

class BackpropagationLearner {
public:
    BackpropagationLearner(Network *network);
    BackpropagationLearner(const BackpropagationLearner& orig);
    virtual ~BackpropagationLearner();
    // Launches the learning process.
    void train(LabeledDataset *dataset);
private:
    // Validates input dataset provided for learning against neural network.
    void validate(LabeledDataset *dataset);
    // We cannot reuse the forward run from the network's implementation,
    // because additional meta results need to be kept for backpropagation
    // algorithm.
    void doForwardPhase(float *input);
    // Backward phase optimizing network parameters in the learning process.
    void doBackwardPhase(float *expectedOutput);
    // Computes total differential for weights in output layer.
    void computeOutputLayer(float *expectedOutput);
    // Computes total differential for weights in hidden layers.
    void computeHiddenLayers();
    // Adjust network weights according to computed total differentials.
    void adjustWeights();
    // Helper method for clearing network layer.
    void clearLayer(float *inputPtr, int layerSize);
    // Allocates memory for caching variables.
    void allocateCache();
    // ANN itself. Used for accessing configuration and tuning weights.
    Network *network;
    // Learning parameter. Intended to be decreasing during learning process.
    float learningRate;
    // Represents average error of the current network configuration.
    float errorTotal;
    // Counter of epochs, incremented right before new epoch is started.
    int epochCounter;
    // Total differential for weight adjustment.
    float *weightDiffs;
    // Cache for local gradients of respective neurons.
    float *localGradients;
};

#endif	/* BACKPROPAGATIONLEARNER_H */

