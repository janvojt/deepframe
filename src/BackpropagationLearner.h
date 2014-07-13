/* 
 * File:   BackpropagationLearner.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 12:10 AM
 */

#ifndef BACKPROPAGATIONLEARNER_H
#define	BACKPROPAGATIONLEARNER_H

#include "Network.h"
#include <limits>

class BackpropagationLearner {
public:
    BackpropagationLearner(Network *network);
    BackpropagationLearner(const BackpropagationLearner& orig);
    virtual ~BackpropagationLearner();
    // Launches the learning process.
    void learn();
private:
    void initInputs();
    void initPotentials();
    void clearLayer(float *inputPtr, int layerSize);
    // ANN itself. Used for accessing configuration and tuning weights.
    Network *network;
    // Array representing the potential coming into each neuron.
    float *inputs;
    // Array representing the potential coming into each neuron before it is
    // processed by activation function.
    float *potentials;
    // Learning parameter. Intended to be decreasing during learning process.
    float learningRate;
    // Represents average error of the current network configuration.
    float errorTotal;
    // Counter of epochs, incremented right before new epoch is started.
    int epochCounter;
};

#endif	/* BACKPROPAGATIONLEARNER_H */

