/* 
 * File:   Network.h
 * Author: janvojt
 *
 * Created on May 30, 2014, 12:17 AM
 */

#ifndef NETWORK_H
#define	NETWORK_H

#include "NetworkConfiguration.h"


class Network {
public:
    Network(NetworkConfiguration* conf);
    Network(const Network& orig);
    virtual ~Network();
    // Returns the network configuration.
    NetworkConfiguration* getConfiguration();
    // Sets the input values for the network.
    // Size of given input array should be equal to the number of input neurons.
    void setInput(float* input);
private:
    // initialize network weights
    void initWeights();
    // initialize input potential for neurons
    void initInputs();
    // Number of layers in the network.
    int noLayers;
    // Network configuration.
    NetworkConfiguration* conf;
    // Array representing weights for each edge in the neural network.
    // The zero-layer weights are for edges coming into input neurons,
    // therefore always initialized to 1.
    float* weights;
    // Array representing the potential coming into each neuron.
    // The potential coming into input neurons is also represented
    // and set when #setInput is called.
    float* inputs;
    // Network bias.
    float bias;
};

#endif	/* NETWORK_H */

