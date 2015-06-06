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

using namespace std;

class Layer {
    
public:
    Layer();
    Layer(const Layer& orig);
    virtual ~Layer();
    
    virtual void setup(Layer *previousLayer, string confString) = 0;

    void forward();
    virtual void forwardCpu() = 0;
    virtual void forwardGpu() = 0;
    
    void backward();
    virtual void backwardCpu() = 0;
    virtual void backwardGpu() = 0;
    
    void setUseGpu(bool useGpu);
    
    int getWeightsCount();
    
    int getOutputsCount();
    
    data_t *getInputs();
    void setInputs(data_t *inputs);
    
    data_t *getWeights();
    void setWeights(data_t *weights);
    
    void setNextLayer(Layer *nextLayer);
    
protected:
    data_t *inputs = NULL;
    
    int inputsCount;
    
    data_t *weights = NULL;
    
    int weightsCount;

    Layer *previousLayer = NULL;

    Layer *nextLayer = NULL;
    
    bool isLast = true;
    
private:
    bool useGpu = false;
};

#endif	/* LAYER_H */

