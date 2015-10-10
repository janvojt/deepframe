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
#include "../../util/cudaHelpers.h"

using namespace std;

struct ConvolutionalConfig {
    int windowSize;
    int featureMultiplier;
    bool useBias;
};

class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer& orig);
    virtual ~ConvolutionalLayer();

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();
    
    virtual void backwardLastCpu(data_t* expectedOutput);
    virtual void backwardLastGpu(data_t* expectedOutput);
    
    ConvolutionalConfig getConfig();
    
    int getOutputFeatures();
    
    int getOutputWidth();
    
    int getOutputHeight();
    
protected:
    
    void setup(string confString);
    
private:
    
    void processConfString(string confString);
    
    void k_weightGemm(const data_t* input,
        const data_t* output, data_t* weights);
    
    void k_backwardGemm(const data_t* output,
    const data_t* weights, data_t* input);

    inline void k_conv_im2col(const data_t* data, data_t* colBuffer) {
        k_im2col(data, inputFeatures, inputFeatureHeight, inputFeatureWidth,
                kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, colBuffer);
    }

    inline void k_conv_col2im(const data_t* colBuffer, data_t* data) {
        k_col2im(colBuffer, inputFeatures, inputFeatureHeight, inputFeatureWidth,
                kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, data);
    }
    
    ConvolutionalConfig conf;
    
    data_t *colBuffer;

    int featuresCount;
    int inputFeatures;
    int featureSize;
    int featureWidth;
    int featureHeight;
    int inputFeatureHeight;
    int inputFeatureWidth;
    
    int colWidth;
    int colHeight;
    int colSize;
    
    int kernelHeight, kernelWidth;
    int kernelDim;
    
    // TODO implement stride and padding
    int strideHeight = 1, strideWidth = 1;
    int padHeight = 0, padWidth = 0;
};

#endif	/* CONVOLUTIONALLAYER_H */

