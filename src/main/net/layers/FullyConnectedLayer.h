/* 
 * File:   FullyConnectedLayer.h
 * Author: janvojt
 *
 * Created on May 17, 2015, 12:55 AM
 */

#ifndef FULLYCONNECTEDLAYER_H
#define	FULLYCONNECTEDLAYER_H

#include "../Layer.h"

template <typename dType>
struct FullyConnectedConfig {
    
    int outputSize;
    
    bool useBias;
    
    /**
     * Pointer to activation function normalizing the neurons potential.
     * Input potential is preserved and the normalized value
     * is put into the target array. It is also possible to provide
     * the same pointer for input and target for in-place computation
     * saving some memory.
     */
    void (*activationFnc)(dType *x, dType *y, int layerSize);
    
    /** Derivative of activation function. */
    void (*dActivationFnc)(dType *x, dType *y, int layerSize);
};

template <typename dType>
class FullyConnectedLayer : public Layer<dType> {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(const FullyConnectedLayer& orig);
    virtual ~FullyConnectedLayer();
    
    void setup(Layer<dType> *previousLayer, FullyConnectedConfig<dType> conf);

    void forward();
    
private:
    FullyConnectedConfig<dType> conf;
};

#endif	/* FULLYCONNECTEDLAYER_H */

