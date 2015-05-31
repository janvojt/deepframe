/* 
 * File:   GpuBackpropagationLearner.h
 * Author: janvojt
 *
 * Created on November 20, 2014, 11:23 PM
 */

#ifndef GPUBACKPROPAGATIONLEARNER_H
#define	GPUBACKPROPAGATIONLEARNER_H

#include "../common.h"

#include "BackpropagationLearner.h"
#include "../net/GpuNetwork.h"

class GpuBackpropagationLearner : public BackpropagationLearner {
public:
    GpuBackpropagationLearner(GpuNetwork * network);
    GpuBackpropagationLearner(const GpuBackpropagationLearner& orig);
    virtual ~GpuBackpropagationLearner();
    // Computes local gradients for output neurons.
    void computeOutputGradients(data_t *expectedOutput);
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

#endif	/* GPUBACKPROPAGATIONLEARNER_H */

