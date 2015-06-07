/* 
 * File:   CpuNetwork.h
 * Author: janvojt
 *
 * Created on November 8, 2014, 12:48 AM
 */

#ifndef CPUNETWORK_H
#define	CPUNETWORK_H

#include "Network.h"
#include "../common.h"

class CpuNetwork : public Network {
public:
    CpuNetwork(NetworkConfiguration *conf);
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
    
    virtual void setExpectedOutput(data_t* output);
    
    virtual void forward();

    virtual void backward();
    
protected:

    void allocateMemory();
    
private:
    
    data_t *expectedOutput;
};

#endif	/* CPUNETWORK_H */

