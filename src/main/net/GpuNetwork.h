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

#include <iostream>

#include "../common.h"
#include "../util/cudaHelpers.h"

class GpuNetwork : public Network {
public:
    GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf);
    GpuNetwork(const GpuNetwork& orig);
    virtual ~GpuNetwork();
    
    /**
     * Creates a network clone.
     *   
     * @return network clone with copied weights, potentials, bias, etc.
     */
    GpuNetwork *clone();
    
    /**
     * Merges weights and bias from given networks into this network.
     * 
     * @param nets array of networks to be merged into this network
     * @param size number of networks in given array
     */
    void merge(Network **nets, int size);
    
    /**
     * Reinitializes network so it forgets everything it learnt.
     * This means random reinitialization of weights and bias.
     */
    void reinit();
    
    /**
     * Sets the input values for the network.
     * Size of given input array should be equal to the number of input neurons.
     */
    void setInput(data_t *input);
    
    /** Returns pointer to the beginning of the input array. */
    data_t *getInput();
    /** Returns pointer to the beginning of the output array. */
    data_t *getOutput();

    virtual void setExpectedOutput(data_t* output);
    
    /**
     * Performs the forward run on the network. Used in training and testing.
     */
    virtual void forward();

    /**
     * Performs the backward run on the network. Used in training.
     */
    virtual void backward();
    
protected:

    /**
     * Allocates additional memory needed by the network.
     * Launched in the network initialization phase.
     */
    void allocateMemory();
    
private:
    
    /**
     * Array representing input coming into the network.
     * Allocated on host memory.
     */
    data_t *input;
    
    /**
     * Tells whether network input is synchronized
     * between GPU and CPU.
     */
    bool inputSynced = false;
    
    /**
     * Array representing network output.
     * Allocated on host memory.
     */
    data_t *output;
    
    /**
     * Tells whether network output is synchronized
     * between CPU and GPU.
     */
    bool outputSynced = false;
    
    /** Expected output of the network for current input (pattern). */
    data_t *expectedOutput;
    
    /** Memory size required to store #expectedOutput. */
    int memExpectedOutput;
    
    /** Holds GPU configuration and device properties. */
    GpuConfiguration *gpuConf;
    
    /** CUDA Basic Linear Algebra Subprograms handle. */
    cublasHandle_t cublasHandle;
};

#endif	/* GPUNETWORK_H */

