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
    // Builds artificial neural network from network configuration
    // given in the constructor argument.
    Network(NetworkConfiguration *conf);
    Network(const Network& orig);
    virtual ~Network();
    
    /** Creates a network clone.
        
        @return network clone with copied weights, potentials, bias, etc.
     */
    virtual Network *clone() = 0;
    
    /**
     * Merges weights and bias from given networks into this network.
     * 
     * @param nets array of networks to be merged into this network
     * @param size number of networks in given array
     */
    virtual void merge(Network **nets, int size) = 0;
    
    /** Returns the network configuration.
     */
    NetworkConfiguration *getConfiguration();
    
    /** Reinitializes network so it forgets everything it learnt.

        This usually means random reinitialization of weights and bias.
     */
    virtual void reinit() = 0;
    
    // run the network
    virtual void run() = 0;
    // Sets the input values for the network.
    // Size of given input array should be equal to the number of input neurons.
    virtual void setInput(double *input) = 0;
    // Returns pointer to the beginning of array with neuron inputs
    // (potential after being processed by the activation function).
    // Values at the beginning actually belong to the input layer. Activation
    // function is not applied to these, therefore they can represent original
    // network input.
    virtual double *getInputs() = 0;
    // Returns pointer to the beginning of the input array.
    virtual double *getInput() = 0;
    // Returns pointer to the beginning of the output array.
    virtual double *getOutput() = 0;
    // Returns number of neurons in the first layer.
    int getInputNeurons();
    // Returns number of neurons in the last layer.
    int getOutputNeurons();
    // Returns the total number of all neurons in all layers.
    virtual int getAllNeurons() = 0;
    // Returns offset where the input array index starts for given layer.
    // Input layer has index zero, while its returned offset is also zero.
    // Therefore offset for the output layer can be obtained by asking
    // for layer index (number of layers - 1). Furthermore, if number of layers
    // is provided as layer index, number of all neurons in the net is returned.
    virtual int getInputOffset(int layer) = 0;
    // Returns pointer to the beginning of array with weights
    // for neuron connections.
    // This internal network property is usually needed
    // in the process of learning.
    virtual double *getWeights() = 0;
    // Returns offset where the weight array index starts for weights between
    // given layer and the preceeding layer.
    // Input layer has index zero, while its returned offset is also zero.
    // Therefore offset for the output layer can be obtained by asking
    // for layer index (number of layers - 1). Furthermore, if number of layers
    // is provided as layer index, number of all weights in the net is returned.
    virtual int getWeightsOffset(int layer) = 0;
    // Provides access to bias values,
    // so the learning algorithm may adjust them.
    virtual double *getBiasValues() = 0;
protected:
    // Network configuration.
    NetworkConfiguration *conf;
    // Number of layers in the network.
    int noLayers;
    
};

#endif	/* NETWORK_H */

