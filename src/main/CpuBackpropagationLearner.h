/* 
 * File:   CpuBackpropagationLearner.h
 * Author: janvojt
 *
 * Created on November 20, 2014, 9:40 PM
 */

#ifndef CPUBACKPROPAGATIONLEARNER_H
#define	CPUBACKPROPAGATIONLEARNER_H

#include "BackpropagationLearner.h"
#include "CpuNetwork.h"


class CpuBackpropagationLearner : public BackpropagationLearner {
public:
    CpuBackpropagationLearner(CpuNetwork *network);
    CpuBackpropagationLearner(const CpuBackpropagationLearner& orig);
    virtual ~CpuBackpropagationLearner();
protected:
    // Computes local gradients for output neurons.
    void computeOutputGradients(double *expectedOutput);
    // Computes total differential for all weights
    // and local gradients for hidden neurons.
    void computeWeightDifferentials();
    // Adjust network weights according to computed total differentials.
    void adjustWeights();
    // Adjust network bias according to computed total differentials.
    void adjustBias();
private:
    // Allocates memory for caching variables.
    void allocateCache();

};

#endif	/* CPUBACKPROPAGATIONLEARNER_H */

