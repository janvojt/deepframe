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
class ConvolutionalLayer;

template <typename dType>
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
    
    void setup(ConvolutionalLayer<dType>* previousLayer, SubsamplingConfig<dType> conf);
    
    /**
     * Set this layer up as an input layer.
     * Such an input layer is represented by a subsampling map with
     * one single feature map.
     * 
     * @param outputWidth input/output width
     * @param outputHeight input.output height
     */
    void setupAsInput(int outputWidth, int outputHeight);

    void forward();
    
    SubsamplingConfig<dType> getConfig();
    
    int getFeatureWidth();
    
    int getFeatureHeight();
    
    int getFeaturesCount();
    
private:
    SubsamplingConfig<dType> conf;
    
    int featuresCount;
    
    int featureWidth;
    
    int featureHeight;
    
    int inputWidth;
    
    int inputHeight;
};

#endif	/* SUBSAMPLINGLAYER_H */

