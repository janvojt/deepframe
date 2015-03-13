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

template <typename dType>
class MseErrorComputer : public ErrorComputer<dType> {
public:
    MseErrorComputer();
    MseErrorComputer(const MseErrorComputer& orig);
    virtual ~MseErrorComputer();
    // Computes the Mean Square Error for given expected output.
    dType compute(Network<dType>* net, dType* expectedOutput);
private:

};

#endif	/* MSEERRORCOMPUTER_H */

