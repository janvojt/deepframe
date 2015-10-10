/* 
 * File:   Layer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 2:19 PM
 */

#ifndef LAYER_H
#define	LAYER_H

#include "../common.h"
#include <string>
#include <cublas_v2.h>
#include "NetworkConfiguration.h"

using namespace std;

class Layer {
    
public:
    Layer();
    Layer(const Layer& orig);
    virtual ~Layer();
    
    void setup(Layer *previousLayer, NetworkConfiguration *netConf, string confString);

    virtual void forwardCpu() = 0;
    virtual void forwardGpu() = 0;
    
    virtual void backwardCpu() = 0;
    virtual void backwardGpu() = 0;
    
    virtual void backwardLastCpu(data_t *expectedOutput) = 0;
    virtual void backwardLastGpu(data_t *expectedOutput) = 0;
    
    int getWeightsCount();
    
    int getOutputsCount();
    
    data_t *getOutputs();
    data_t *getOutputDiffs();
    void setInputs(data_t *inputs, data_t *outputDiffs);
    
    data_t *getWeights();
    data_t *getWeightDiffs();
    void setWeights(data_t *weights, data_t *weightDiffs);
    
    void setNextLayer(Layer *nextLayer);
    
    bool isFirst();
    bool isLast();
    
    // CUDA Basic Linear Algebra Subprograms handle.
    cublasHandle_t cublasHandle;
    
protected:
    
    virtual void setup(string confString) = 0;
    
    data_t *outputs = NULL;
    data_t *outputDiffs = NULL;
    
    int outputsCount;
    
    data_t *weights = NULL;
    
    data_t *weightDiffs = NULL;
    
    int weightsCount;

    Layer *previousLayer = NULL;

    Layer *nextLayer = NULL;
    
    bool first = true;
    bool last = true;
    
    NetworkConfiguration *netConf;
    
    /** Learning rate. */
    data_t lr;
    
};

#endif	/* LAYER_H */

