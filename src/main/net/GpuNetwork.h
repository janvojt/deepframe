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
    
    /** Creates a network clone.
        
        @return network clone with copied weights, potentials, bias, etc.
     */
    GpuNetwork *clone();
    
    /**
     * Merges weights and bias from given networks into this network.
     * 
     * @param nets array of networks to be merged into this network
     * @param size number of networks in given array
     */
    void merge(Network **nets, int size);
    
    /** Reinitializes network so it forgets everything it learnt.

        This means random reinitialization of weights and bias.
     */
    void reinit();
    
    // Sets the input values for the network.
    // Size of given input array should be equal to the number of input neurons.
    void setInput(data_t *input);
    // Returns pointer to the beginning of the input array.
    data_t *getInput();
    // Returns pointer to the beginning of the output array.
    data_t *getOutput();
    
    virtual void forward();

    virtual void backward();
    
protected:

    void allocateMemory();
    
private:
    // Array representing input coming into the network.
    // Allocated on host memory.
    data_t *input;
    // Array representing network output.
    // Allocated on host memory.
    data_t *output;
    // Holds GPU configuration and device properties.
    GpuConfiguration *gpuConf;
    // CUDA Basic Linear Algebra Subprograms handle.
    cublasHandle_t cublasHandle;
};

#endif	/* GPUNETWORK_H */

