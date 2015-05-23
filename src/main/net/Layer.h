/* 
 * File:   Layer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 2:19 PM
 */

#ifndef LAYER_H
#define	LAYER_H

template <typename dType>
class Layer {
    
public:
    Layer();
    Layer(const Layer& orig);
    virtual ~Layer();

    virtual void forward() = 0;
    
//    virtual void backward() = 0;
    
    int getWeightsCount();
    
    int getOutputsCount();
    
    dType *getInputs();
    void setInputs(dType *inputs);
    
    dType *getWeights();
    void setWeights(dType *weights);
    
    void setNextLayer(Layer *nextLayer);
    
protected:
    dType *inputs;
    
    int inputsCount;
    
    dType *weights;
    
    int weightsCount;

    Layer<dType> *previousLayer;
    
    Layer<dType> *nextLayer;
    
    bool isLast = true;
};

#endif	/* LAYER_H */

