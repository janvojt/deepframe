/* 
 * File:   MseErrorComputer.h
 * Author: janvojt
 *
 * Created on July 20, 2014, 11:48 AM
 */

#ifndef MSEERRORCOMPUTER_H
#define	MSEERRORCOMPUTER_H

#include "ErrorComputer.h"

class MseErrorComputer : public ErrorComputer {
public:
    MseErrorComputer();
    MseErrorComputer(const MseErrorComputer& orig);
    virtual ~MseErrorComputer();
    // Computes the Mean Square Error for given expected output.
    float compute(Network* net, float* expectedOutput);
private:

};

#endif	/* MSEERRORCOMPUTER_H */

