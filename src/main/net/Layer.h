/* 
 * File:   Layer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 2:19 PM
 */

#ifndef LAYER_H
#define	LAYER_H

#include "../common.h"
#include <string>
#include <cublas_v2.h>
#include <curand.h>
#include "NetworkConfiguration.h"

using namespace std;

/**
 * Represents one layer in the artificial neural network.
 */
class Layer {
    
public:
    Layer();
    Layer(const Layer& orig);
    virtual ~Layer();
    
    /**
     * Initializes the layer, including its configurations and pointer to preceding layer.
     * 
     * @param previousLayer the preceding layer
     * @param netConf network configuration
     * @param confString configuration string for the layer
     */
    void setup(Layer *previousLayer, NetworkConfiguration *netConf, string confString);

    /**
     * Propagates the signals into the following layer using CPU.
     */
    virtual void forwardCpu() = 0;
    
    /**
     * Propagates the signals into the following layer using GPU.
     */
    virtual void forwardGpu() = 0;
    
    /**
     * Backpropagates the signals and performs parameter updates during the training on CPU.
     */
    virtual void backwardCpu() = 0;
    
    /**
     * Backpropagates the signals and performs parameter updates during the training on GPU.
     */
    virtual void backwardGpu() = 0;
    
    /**
     * Backpropagates the signals and performs parameter updates based on expected output on CPU.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastCpu(data_t *expectedOutput) = 0;
    
    /**
     * Backpropagates the signals and performs parameter updates based on expected output on GPU.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastGpu(data_t *expectedOutput) = 0;
    
    /**
     * @return number of trainable parameters for this layers
     */
    int getWeightsCount();
    
    /**
     * @return number of output neurons in this layer
     */
    int getOutputsCount();
    
    /**
     * @return output neuron activations
     */
    data_t *getOutputs();
    
    /**
     * @return gradients for activations in output neurons
     */
    data_t *getOutputDiffs();
    
    /**
     * Initializes pointers for the neuron activations inside the layer.
     * Used during network setup.
     * 
     * @param inputs
     * @param outputDiffs
     */
    void setInputs(data_t *inputs, data_t *outputDiffs);
    
    /**
     * @return trainable parameters for this layer
     */
    data_t *getWeights();
    
    /**
     * @return the updates for the trainable parameters in this layer
     */
    data_t *getWeightDiffs();
    
    /**
     * Initializes pointers for trainable parameters in this layer.
     * Used during network setup.
     * 
     * @param weights
     * @param weightDiffs
     */
    void setWeights(data_t *weights, data_t *weightDiffs);
    
    /**
     * Set the following layer.
     * 
     * @param nextLayer the following layer
     */
    void setNextLayer(Layer *nextLayer);
    
    /**
     * @return whether this is the input layer of the network
     */
    bool isFirst();
    
    /**
     * @return whether this is the output layer of the network
     */
    bool isLast();
    
    /**
     * @return whether this layer is pretrainable
     */
    virtual bool isPretrainable();
    
    /**
     * Implements the pretraining on CPU.
     */
    virtual void pretrainCpu();
    
    /**
     * Implements the pretraining on GPU.
     */
    virtual void pretrainGpu();
    
    /** CUDA Basic Linear Algebra Subprograms handle. */
    cublasHandle_t cublasHandle;
    
    /** Random generator for cuRAND. */
    curandGenerator_t *curandGen;

    /** Memory for holding random data used by the pretrainer. */
    data_t *randomData;
    
protected:
    
    virtual void setup(string confString) = 0;
    
    /** Activations in the output neurons of this layer. */
    data_t *outputs = NULL;
    
    /** Gradients for output neurons. */
    data_t *outputDiffs = NULL;
    
    /** Number of output neurons in this layer. */
    int outputsCount;
    
    /** Trainable parameters in this layer. */
    data_t *weights = NULL;
    
    /** Computed updates for the trainable parameters. */
    data_t *weightDiffs = NULL;
    
    /** Number of weights including bias. */
    int weightsCount;
    
    /** Number of weights excluding bias. */
    int genuineWeightsCount;

    /** Pointer to the preceding layer. */
    Layer *previousLayer = NULL;

    /** Pointer to the next layer. */
    Layer *nextLayer = NULL;
    
    bool first = true;
    bool last = true;
    
    NetworkConfiguration *netConf;
    
    /** Learning rate. */
    data_t lr;
    
};

#endif	/* LAYER_H */

