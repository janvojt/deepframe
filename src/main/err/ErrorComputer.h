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

class ErrorComputer {
public:
    ErrorComputer();
    ErrorComputer(const ErrorComputer& orig);
    virtual ~ErrorComputer();
    // Computes the error for given expected output.
    virtual data_t compute(Network *net, data_t *expectedOutput) = 0;
private:

};

#endif	/* ERRORCOMPUTER_H */

