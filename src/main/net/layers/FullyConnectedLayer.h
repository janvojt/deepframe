/* 
 * File:   FullyConnectedLayer.h
 * Author: janvojt
 *
 * Created on May 17, 2015, 12:55 AM
 */

#ifndef FULLYCONNECTEDLAYER_H
#define	FULLYCONNECTEDLAYER_H

#include "../Layer.h"

#include <string>

#include "../../common.h"

using namespace std;

/** Holds the configuration parameters for perceptron layers. */
struct FullyConnectedConfig {
    
    /** Number of output neurons. */
    int outputSize;
    
    /** Use bias for this layer? */
    bool useBias;
    
};

/**
 * Represents the fully connected layer as a building block of Multilayer Perceptron.
 */
class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(const FullyConnectedLayer& orig);
    virtual ~FullyConnectedLayer();

    /**
     * Performs the forward pass during training and testing on CPU.
     */
    void forwardCpu();
    /**
     * Performs the forward pass during training and testing on GPU.
     */
    void forwardGpu();
    
    /**
     * Performs the backward pass during training on CPU.
     */
    void backwardCpu();
    /**
     * Performs the backward pass during training on GPU.
     */
    void backwardGpu();
    
    /**
     * Propagates the signals down to the preceding layer
     * and adjusts the network parameters with respect to
     * expected output using CPU.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastCpu(data_t* expectedOutput);
    /**
     * Propagates the signals down to the preceding layer
     * and adjusts the network parameters with respect to
     * expected output using GPU.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastGpu(data_t* expectedOutput);
    
protected:
    
    /**
     * Initializes the layer to the correct state given by the configuration.
     * 
     * @param confString layer configuration string
     */
    void setup(string confString);

private:
    
    /**
     * Parses the configuration string.
     * 
     * @param confString
     */
    void processConfString(string confString);
    
    /** Configuration for this layer. */
    FullyConnectedConfig conf;
    
    /**
     *  Cache for storing meta-results when computing gradients
     *  for neuron signals with respect to error.
     */
    data_t *thisOutputDerivatives = NULL;
};

#endif	/* FULLYCONNECTEDLAYER_H */

