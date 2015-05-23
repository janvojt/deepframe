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
    
    // Returns pointer to the beginning of the input array.
    dType *getInput();
    
    // Returns pointer to the beginning of the output array.
    dType *getOutput();
    
protected:

    void allocateMemory();
    
private:
    
};

#endif	/* CPUNETWORK_H */

