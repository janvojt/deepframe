/* 
 * File:   ErrorComputer.h
 * Author: janvojt
 *
 * Created on July 20, 2014, 11:46 AM
 */

#ifndef ERRORCOMPUTER_H
#define	ERRORCOMPUTER_H

#include "../net/Network.h"

template <typename dType>
class ErrorComputer {
public:
    ErrorComputer();
    ErrorComputer(const ErrorComputer& orig);
    virtual ~ErrorComputer();
    // Computes the error for given expected output.
    virtual dType compute(Network<dType> *net, dType *expectedOutput) = 0;
private:

};

#endif	/* ERRORCOMPUTER_H */

