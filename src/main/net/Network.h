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
    
    /**
     * Builds artificial neural network from network configuration
     * given in the constructor argument.
     * 
     * @param conf
     */
    Network(NetworkConfiguration *conf);
    Network(const Network& orig);
    virtual ~Network();
    
    /**
     * Creates a network clone.
     * 
     * @return network clone with copied weights, potentials, bias, etc.
     */
    virtual Network *clone() = 0;
    
    /**
     * Merges weights and bias from given networks into this network.
     * 
     * @param nets array of networks to be merged into this network
     * @param size number of networks in given array
     */
    virtual void merge(Network **nets, int size) = 0;
    
    /**
     * Returns the network configuration.
     */
    NetworkConfiguration *getConfiguration();
    
    /**
     * Reinitializes network so it forgets everything it learnt.
     * This usually means random reinitialization of weights and bias.
     */
    virtual void reinit() = 0;
    
    /** Run the network. */
    virtual void forward() = 0;
    
    /** Backpropagate errors. */
    virtual void backward() = 0;
       
    /**
     * Size of given input array should be equal to the number of input neurons.
     * 
     * @param input the input values for the network.
     */
    virtual void setInput(data_t *input) = 0;
    
    /**
     * Values at the beginning actually belong to the input layer. Activation
     * function is not applied to these, therefore they can represent original
     * network input.
     * 
     * @return pointer to the beginning of array with neuron inputs
     *  (potential after being processed by the activation function).
     */
    data_t *getInputs();

    /**
     * @return pointer to the beginning of the input array.
     */
    virtual data_t *getInput() = 0;
    
    /**
     * @return pointer to the beginning of the output array.
     */
    virtual data_t *getOutput() = 0;
    
    virtual void setExpectedOutput(data_t *output) = 0;
    
    /**
     * @return number of neurons in the first layer.
     */
    int getInputNeurons();
    
    /**
     * @return number of neurons in the last layer.
     */
    int getOutputNeurons();
    
    /**
     * @return the total number of all neurons in all layers.
     */
    int getInputsCount();
    
    /**
     * @return pointer to the beginning of array with weights
     *  for neuron connections.
     */
    data_t *getWeights();
    
    /**
     * @return number of trainable parameters in this network
     */
    int getWeightsCount();
    
    /**
     * Network initialization, which also triggers layer initialization.
     */
    void setup();
    
    /**
     * Add a new layer into the network, put on the top of already added layers.
     * 
     * @param layer network layer to be added
     */
    void addLayer(Layer *layer);
    
    /**
     * Get the layer specified by given index. Input layer
     * has an index of zero.
     * 
     * @param index layer index
     * @return the layer
     */
    Layer *getLayer(int index);

    /**
     * Exports network parameters into external file.
     *
     * @param filPath
     */
    virtual void save(char *filePath) = 0;
    
protected:
    
    /**
     * Allocates the memory required for the network. Includes
     * neural activations, potentials, trainable parameters,
     * gradients, etc.
     */
    virtual void allocateMemory() = 0;
    
    /**
     * Performs any input preprocessing necessary for given network type.
     * 
     * @param input network input pattern
     */
    void processInput(data_t *input);
    
    /** Network configuration. */
    NetworkConfiguration *conf;
    
    /** Number of layers in the network. */
    int noLayers;
    
    /** An array of pointers pointing to the layers within this network. */
    Layer **layers;
    
    /** Network input signals (observed environment). */
    data_t *inputs;
    
    /** Gradients for output activations. */
    data_t *outputDiffs;

    /** Number of network inputs. */
    int inputsCount = 0;
    
    /** Trainable parameters of the network. */
    data_t *weights;
    
    /** The computed updates for trainable parameters in this network. */
    data_t *weightDiffs;
    
    /** Number of ttrainable parameters in this network. */
    int weightsCount = 0;
    
private:
    
    /** Cursor for adding new #layers. */
    int layerCursor = 0;
    
    /** Has this network been initialized already? */
    bool isInitialized = false;
};

#endif	/* NETWORK_H */

