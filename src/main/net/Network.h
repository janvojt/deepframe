/* 
 * File:   Network.h
 * Author: janvojt
 *
 * Created on May 30, 2014, 12:17 AM
 */

#ifndef NETWORK_H
#define	NETWORK_H

#include "NetworkConfiguration.h"
#include "Layer.h"
#include "../common.h"

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
    
    /** Run the network. */
    void forward();
    
    /** Backpropagate errors. */
    void backward();
       
    // Sets the input values for the network.
    // Size of given input array should be equal to the number of input neurons.
    virtual void setInput(data_t *input) = 0;
    // Returns pointer to the beginning of array with neuron inputs
    // (potential after being processed by the activation function).
    // Values at the beginning actually belong to the input layer. Activation
    // function is not applied to these, therefore they can represent original
    // network input.
    data_t *getInputs();
    // Returns pointer to the beginning of the input array.
    virtual data_t *getInput() = 0;
    // Returns pointer to the beginning of the output array.
    virtual data_t *getOutput() = 0;
    // Returns number of neurons in the first layer.
    int getInputNeurons();
    // Returns number of neurons in the last layer.
    int getOutputNeurons();
    // Returns the total number of all neurons in all layers.
    int getInputsCount();
    // Returns pointer to the beginning of array with weights
    // for neuron connections.
    // This internal network property is usually needed
    // in the process of learning.
    data_t *getWeights();
    
    int getWeightsCount();
    
    void setup();
    
    virtual bool useGpu() = 0;
    
    void addLayer(Layer *layer);
    
    Layer *getLayer(int index);
    
protected:
    
    virtual void allocateMemory() = 0;
    
    // Network configuration.
    NetworkConfiguration *conf;
    // Number of layers in the network.
    int noLayers;
    
    Layer **layers;
    
    data_t *inputs;

    int inputsCount = 0;
    
    data_t *weights;
    data_t *weightDiffs;
    
    int weightsCount = 0;
    
private:
    
    /** Cursor for adding new #layers. */
    int layerCursor = 0;
    
    bool isInitialized = false;
};

#endif	/* NETWORK_H */

