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
    
    virtual int getWeightCount() = 0;
    
    virtual int getOutputCount() = 0;
    
    dType *getInputs();
    void setInputs(dType *inputs);
    
    dType *getWeights();
    void setWeights(dType *weights);
    
    void setNextLayer(Layer *nextLayer);
    
protected:
    dType *inputs;
    
    dType *weights;

    Layer<dType> *previousLayer;
    
    Layer<dType> *nextLayer;
    
    bool isLast = true;
};

#endif	/* LAYER_H */

