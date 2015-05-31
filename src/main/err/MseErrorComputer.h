/* 
 * File:   MseErrorComputer.h
 * Author: janvojt
 *
 * Created on July 20, 2014, 11:48 AM
 */

#ifndef MSEERRORCOMPUTER_H
#define	MSEERRORCOMPUTER_H

#include "ErrorComputer.h"
#include "../net/Network.h"
#include "../common.h"

class MseErrorComputer : public ErrorComputer {
public:
    MseErrorComputer();
    MseErrorComputer(const MseErrorComputer& orig);
    virtual ~MseErrorComputer();
    // Computes the Mean Square Error for given expected output.
    data_t compute(Network* net, data_t* expectedOutput);
private:

};

#endif	/* MSEERRORCOMPUTER_H */

