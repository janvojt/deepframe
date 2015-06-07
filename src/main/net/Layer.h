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
    
    int getWeightsCount();
    
    int getOutputsCount();
    
    data_t *getInputs();
    void setInputs(data_t *inputs);
    
    data_t *getWeights();
    void setWeights(data_t *weights, data_t *weightDiffs);
    
    void setNextLayer(Layer *nextLayer);
    
protected:
    
    virtual void setup(string confString) = 0;
    
    data_t *inputs = NULL;
    
    int inputsCount;
    
    data_t *weights = NULL;
    
    data_t *weightDiffs = NULL;
    
    int weightsCount;

    Layer *previousLayer = NULL;

    Layer *nextLayer = NULL;
    
    bool isLast = true;
    
    NetworkConfiguration *netConf;
};

#endif	/* LAYER_H */

