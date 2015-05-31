/* 
 * File:   FullyConnectedLayer.h
 * Author: janvojt
 *
 * Created on May 17, 2015, 12:55 AM
 */

#ifndef FULLYCONNECTEDLAYER_H
#define	FULLYCONNECTEDLAYER_H

#include "../Layer.h"
#include "../LayerFactory.h"
#include "../../common.h"

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
    void (*activationFnc)(data_t *x, data_t *y, int layerSize);
    
    /** Derivative of activation function. */
    void (*dActivationFnc)(data_t *x, data_t *y, int layerSize);
};

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(const FullyConnectedLayer& orig);
    virtual ~FullyConnectedLayer();
    
    void setup(Layer *previousLayer, FullyConnectedConfig conf);

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();

private:
    FullyConnectedConfig conf;
};

#endif	/* FULLYCONNECTEDLAYER_H */

