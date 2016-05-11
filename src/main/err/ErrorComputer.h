/* 
 * File:   ErrorComputer.h
 * Author: janvojt
 *
 * Created on July 20, 2014, 11:46 AM
 */

#ifndef ERRORCOMPUTER_H
#define	ERRORCOMPUTER_H

#include "../net/Network.h"
#include "../common.h"

/**
 * Represents an interface for different implementations of determining
 * the error rate of the network with respect to expected output.
 */
class ErrorComputer {
public:
    ErrorComputer();
    ErrorComputer(const ErrorComputer& orig);
    virtual ~ErrorComputer();
    
    /**
     * Computes the error for given expected output.
     * 
     * @param net the neural network
     * @param expectedOutput expected output
     * @return the error computed
     */
    virtual data_t compute(Network *net, data_t *expectedOutput) = 0;
private:

};

#endif	/* ERRORCOMPUTER_H */

