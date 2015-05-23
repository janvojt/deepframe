/* 
 * File:   SubsamplingLayer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 11:38 PM
 */

#ifndef SUBSAMPLINGLAYER_H
#define	SUBSAMPLINGLAYER_H

#include "ConvolutionalLayer.h"

template <typename dType>
struct SubsamplingConfig {
    int inputFeatures;
    int inputWidth;
    int inputHeight;
    int windowWidth;
    int windowHeight;
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
class SubsamplingLayer : public Layer<dType> {
public:
    SubsamplingLayer();
    SubsamplingLayer(const SubsamplingLayer& orig);
    virtual ~SubsamplingLayer();
    
    void setup(Layer<dType>* previousLayer, SubsamplingConfig<dType> conf);

    void forward();
    
    int getWeightCount();
    
    int getOutputCount();
    
private:
    SubsamplingConfig<dType> conf;
};

#endif	/* SUBSAMPLINGLAYER_H */

