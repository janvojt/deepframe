/* 
 * File:   SubsamplingLayer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 11:38 PM
 */

#ifndef SUBSAMPLINGLAYER_H
#define	SUBSAMPLINGLAYER_H

#include "ConvolutionalLayer.h"
#include "../../common.h"

class ConvolutionalLayer;

struct SubsamplingConfig {
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
    void (*activationFnc)(data_t *x, data_t *y, int layerSize);
    
    /** Derivative of activation function. */
    void (*dActivationFnc)(data_t *x, data_t *y, int layerSize);
};

class SubsamplingLayer : public Layer {
public:
    SubsamplingLayer();
    SubsamplingLayer(const SubsamplingLayer& orig);
    virtual ~SubsamplingLayer();
    
    void setup(ConvolutionalLayer* previousLayer, SubsamplingConfig conf);

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();
    
    SubsamplingConfig getConfig();
    
    int getFeatureWidth();
    
    int getFeatureHeight();
    
    int getFeaturesCount();
    
private:
    SubsamplingConfig conf;
    
    int featuresCount;
    
    int featureWidth;
    
    int featureHeight;
    
    int inputWidth;
    
    int inputHeight;
};

#endif	/* SUBSAMPLINGLAYER_H */

