/* 
 * File:   ConvolutionalLayer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 11:38 PM
 */

#ifndef CONVOLUTIONALLAYER_H
#define	CONVOLUTIONALLAYER_H

#include "../Layer.h"
#include "SubsamplingLayer.h"

template <typename dType>
class SubsamplingLayer;

struct ConvolutionalConfig {
    int windowSize;
    int featureMultiplier;
};

template <typename dType>
class ConvolutionalLayer : public Layer<dType> {
public:
    ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer& orig);
    virtual ~ConvolutionalLayer();
    
    void setup(SubsamplingLayer<dType> *previousLayer, ConvolutionalConfig conf);

    void forward();
    
    ConvolutionalConfig getConfig();
    
    int getOutputFeatures();
    
    int getOutputWidth();
    
    int getOutputHeight();
    
private:
    ConvolutionalConfig conf;
    
    int featuresCount;
    
    int inputFeatures;
    
    int featureWidth;
    
    int featureHeight;
};

#endif	/* CONVOLUTIONALLAYER_H */

