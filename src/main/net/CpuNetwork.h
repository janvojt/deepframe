/* 
 * File:   CpuNetwork.h
 * Author: janvojt
 *
 * Created on November 8, 2014, 12:48 AM
 */

#ifndef CPUNETWORK_H
#define	CPUNETWORK_H

#include "Network.h"

template <typename dType>
class CpuNetwork : public Network<dType> {
public:
    CpuNetwork(NetworkConfiguration<dType> *conf);
    CpuNetwork(const CpuNetwork& orig);
    virtual ~CpuNetwork();
    
    /** Creates a network clone.
        
        @return network clone with copied weights, potentials, bias, etc.
     */
    CpuNetwork *clone();
    
    /**
     * Merges weights and bias from given networks into this network.
     * 
     * @param nets array of networks to be merged into this network
     * @param size number of networks in given array
     */
    void merge(Network<dType> **nets, int size);
    
    /** Reinitializes network so it forgets everything it learnt.

        This means random reinitialization of weights and bias.
     */
    void reinit();
    
    // run the network
    void run();
    // Sets the input values for the network.
    // Size of given input array should be equal to the number of input neurons.
    void setInput(dType *input);
    // Returns pointer to the beginning of array with neuron inputs
    // (potential after being processed by the activation function).
    // Values at the beginning actually belong to the input layer. Activation
    // function is not applied to these, therefore they can represent original
    // network input.
    dType *getInputs();
    // Returns pointer to the beginning of the input array.
    dType *getInput();
    // Returns pointer to the beginning of the output array.
    dType *getOutput();
    // Returns the total number of all neurons in all layers.
    int getAllNeurons();
    // Returns offset where the input array index starts for given layer.
    // Input layer has index zero, while its returned offset is also zero.
    // Therefore offset for the output layer can be obtained by asking
    // for layer index (number of layers - 1). Furthermore, if number of layers
    // is provided as layer index, number of all neurons in the net is returned.
    int getInputOffset(int layer);
    // Returns pointer to the beginning of array with weights
    // for neuron connections.
    // This internal network property is usually needed
    // in the process of learning.
    dType *getWeights();
    // Returns offset where the weight array index starts for weights between
    // given layer and the preceeding layer.
    // Input layer has index zero, while its returned offset is also zero.
    // Therefore offset for the output layer can be obtained by asking
    // for layer index (number of layers - 1). Furthermore, if number of layers
    // is provided as layer index, number of all weights in the net is returned.
    int getWeightsOffset(int layer);
    // Provides access to bias values,
    // so the learning algorithm may adjust them.
    dType *getBiasValues();
    
private:
    // initialize network weights
    void initWeights();
    // initialize input potential for neurons
    void initInputs();
    // Initialize bias if it is enabled in network configuration.
    void initBias();
    // Clears neuron potentials in given layer
    // (zero index represents input layer).
    void clearLayer(dType *inputPtr, int layerSize);
    // Applies bias to layer l (if it is enabled, otherwise does nothing).
    void applyBias(int l);
    // Initializes memory with random doubles from given interval.
    void randomizeDoubles(dType **memPtr, int size);
    // Total number of neurons in the network.
    int noNeurons;
    // Array representing weights for each edge in the neural network.
    // The zero-layer weights are for edges coming into input neurons,
    // therefore always initialized to 1.
    dType *weights;
    // Array representing input coming into each neuron.
    dType *inputs;
    // Network bias. Each neuron has its own bias.
    dType *bias;
    // Cache of number of neurons up to the layer determined by the array index.
    // Used for optimization of calculating indexes for inputs.
    // Method returns zero neurons in zero-th layer.
    int *neuronsUpToLayerCache;
    // Cache of number of weights up to the layer determined by the array index.
    // Used for optimization of calculating indexes for weights.
    // Method returns number of input neurons for first layer.
    // Method further returns number of weights between input and the first
    // hidden layer for layer 2 (weights between first and second layer).
    int *weightsUpToLayerCache;
};

#endif	/* CPUNETWORK_H */

