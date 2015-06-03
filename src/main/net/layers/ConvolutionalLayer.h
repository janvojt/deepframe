/* 
 * File:   ConvolutionalLayer.h
 * Author: janvojt
 *
 * Created on May 16, 2015, 11:38 PM
 */

#ifndef CONVOLUTIONALLAYER_H
#define	CONVOLUTIONALLAYER_H

#include <string>
#include "../Layer.h"
#include "SubsamplingLayer.h"

using namespace std;

struct ConvolutionalConfig {
    int windowSize;
    int featureMultiplier;
};

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer& orig);
    virtual ~ConvolutionalLayer();
    
    void setup(Layer *previousLayer, string confString);

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();
    
    ConvolutionalConfig getConfig();
    
    int getOutputFeatures();
    
    int getOutputWidth();
    
    int getOutputHeight();
    
private:
    
    void processConfString(string confString);
    
    ConvolutionalConfig conf;
    
    int featuresCount;
    
    int inputFeatures;
    
    int featureWidth;
    
    int featureHeight;
};

#endif	/* CONVOLUTIONALLAYER_H */

