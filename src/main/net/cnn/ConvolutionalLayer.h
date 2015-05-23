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

struct ConvolutionalConfig {
    int inputWidth;
    int inputHeight;
    int inputFeatures;
    int windowSize;
    int featureMultiplier;
};

template <typename dType>
class ConvolutionalLayer : public Layer<dType> {
public:
    ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer& orig);
    virtual ~ConvolutionalLayer();
    
    void setup(Layer<dType> *previousLayer, ConvolutionalConfig conf);

    void forward();
    
    int getOutputFeatures();
    
    int getOutputWidth();
    
    int getOutputHeight();
    
private:
    ConvolutionalConfig conf;
};

#endif	/* CONVOLUTIONALLAYER_H */

