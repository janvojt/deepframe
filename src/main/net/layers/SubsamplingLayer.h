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
#include <string>

using namespace std;

class ConvolutionalLayer;

struct SubsamplingConfig {
    int windowWidth;
    int windowHeight;
    bool useBias;
};

class SubsamplingLayer : public Layer {
public:
    SubsamplingLayer();
    SubsamplingLayer(const SubsamplingLayer& orig);
    virtual ~SubsamplingLayer();

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();
    
    virtual void backwardLastCpu(data_t* expectedOutput);
    virtual void backwardLastGpu(data_t* expectedOutput);
    
    SubsamplingConfig getConfig();
    
    int getFeatureWidth();
    
    int getFeatureHeight();
    
    int getFeaturesCount();
    
protected:
    
    void setup(string confString);
    
private:
    
    void processConfString(string confString);
    
    SubsamplingConfig conf;
    
    int featuresCount;
    
    int featureWidth;
    int featureHeight;
    
    int inputFeatureWidth;
    int inputFeatureHeight;
    
    int *maxIndices = NULL;
};

#endif	/* SUBSAMPLINGLAYER_H */

