/* 
 * File:   GpuNetwork.h
 * Author: janvojt
 *
 * Created on November 8, 2014, 12:48 AM
 */

#ifndef GPUNETWORK_H
#define	GPUNETWORK_H

#include "Network.h"
#include "GpuConfiguration.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

class GpuNetwork : public Network {
public:
    GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf);
    GpuNetwork(const GpuNetwork& orig);
    virtual ~GpuNetwork();
    // run the network
    void run();
    // Sets the input values for the network.
    // Size of given input array should be equal to the number of input neurons.
    void setInput(double *input);
    // Returns pointer to the beginning of array with neuron inputs
    // (potential after being processed by the activation function).
    // Values at the beginning actually belong to the input layer. Activation
    // function is not applied to these, therefore they can represent original
    // network input.
    double *getInput();
    // Returns pointer to the beginning of the output array.
    double *getOutput();
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
    double *getWeights();
    // Returns offset where the weight array index starts for weights between
    // given layer and the preceeding layer.
    // Input layer has index zero, while its returned offset is also zero.
    // Therefore offset for the output layer can be obtained by asking
    // for layer index (number of layers - 1). Furthermore, if number of layers
    // is provided as layer index, number of all weights in the net is returned.
    int getWeightsOffset(int layer);
    // Provides access to bias values,
    // so the learning algorithm may adjust them.
    double *getBiasValues();
    
private:
    // initialize network weights
    void initWeights();
    // initialize input potential for neurons
    void initInputs();
    // Initialize bias if it is enabled in network configuration.
    void initBias();
    // Generates random numbers on GPU and copies them to host memory.
    void randomizeDoublesOnGpu(double **hMemory, double **dMemory, int size);
    // Applies bias to layer l (if it is enabled, otherwise does nothing).
    void applyBias(int l);
    // Total number of neurons in the network.
    int noNeurons;
    // Array representing weights for each edge in the neural network.
    // The zero-layer weights are for edges coming into input neurons,
    // therefore always initialized to 1.
    double *weights;
    // Array of weights on GPU device memory.
    double *dWeights;
    // Array representing input coming into each neuron.
    double *inputs;
    double *dInputs;
    // Network bias. Each neuron has its own bias.
    double *bias;
    double *dBias;
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
    // Holds GPU configuration and device properties.
    GpuConfiguration *gpuConf;
    // CUDA Basic Linear Algebra Subprograms handle.
    cublasHandle_t cublasHandle;
};

#endif	/* GPUNETWORK_H */

