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

/** Holds the configuration parameters for this layer. */
struct ConvolutionalConfig {
    int windowWidth;
    int windowHeight;
    int featuresCount;
    bool useBias;
};

/**
 * Represents the convolutional layer as a building block of Convolutional Neural Network.
 */
class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer();
    ConvolutionalLayer(const ConvolutionalLayer& orig);
    virtual ~ConvolutionalLayer();

    /** Propagates the signals up to the next layer using CPU. */
    void forwardCpu();
    /** Propagates the signals up to the next layer using GPU. */
    void forwardGpu();
    
    /**
     * Propagates the signals down to the preceding layer
     * and adjusts the network parameters using CPU.
     */
    void backwardCpu();
    /**
     * Propagates the signals down to the preceding layer
     * and adjusts the network parameters using GPU.
     */
    void backwardGpu();
    
    /**
     * Not implemented for Convolutional layer, as it cannot be the last one.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastCpu(data_t* expectedOutput);
    /**
     * Not implemented for Convolutional layer, as it cannot be the last one.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastGpu(data_t* expectedOutput);
    
    /**
     * @return the configuration of this layer
     */
    ConvolutionalConfig getConfig();
    
    /**
     * @return number of features in this layer
     */
    int getOutputFeatures();
    
    /**
     * @return width of the feature maps in this layer
     */
    int getOutputWidth();
    
    /**
     * @return height of the feature maps in this layer
     */
    int getOutputHeight();
    
protected:
    
    /**
     * Initializes the layer to the correct state given by the configuration.
     * 
     * @param confString layer configuration string
     */
    void setup(string confString);
    
private:
    
    /**
     * Parses the configuration string.
     * 
     * @param confString
     */
    void processConfString(string confString);
    
    /**
     * W = I * O + W
     * 
     * @param input
     * @param output
     * @param weights
     */
    void k_weightGemm(const data_t* input,
        const data_t* output, data_t* weights);
    
    /**
     * I = O * W + I
     * 
     * @param output
     * @param weights
     * @param input
     */
    void k_backwardGemm(const data_t* output,
    const data_t* weights, data_t* input);

    /**
     * Convert image data into column buffer.
     * 
     * @param data image data
     * @param colBuffer converted column buffer
     */
    inline void k_conv_im2col(const data_t* data, data_t* colBuffer) {
        k_im2col(data, inputFeatures, inputFeatureHeight, inputFeatureWidth,
                kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, colBuffer);
    }

    /**
     * Convert column buffer into image.
     * 
     * @param colBuffer column buffer
     * @param data converted image
     */
    inline void k_conv_col2im(const data_t* colBuffer, data_t* data) {
        k_col2im(colBuffer, inputFeatures, inputFeatureHeight, inputFeatureWidth,
                kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, data);
    }
    
    /** Configuration for this layer. */
    ConvolutionalConfig conf;
    
    /** Pointer to the column buffer. */
    data_t *colBuffer;

    /** Number of features in this layer. */
    int featuresCount;
    
    /** Number of features in preceding layer. */
    int inputFeatures;
    
    /** Number of neurons in a feature map. */
    int featureSize;
    
    /** Width of feature maps in this layer. */
    int featureWidth;
    
    /** Height of feature maps in this layer. */
    int featureHeight;
    
    /** Width of feature maps in preceding layer. */
    int inputFeatureWidth;
    
    /** Height of feature maps in preceding layer. */
    int inputFeatureHeight;
    
    // Column buffer size.
    int colWidth;
    int colHeight;
    int colSize;
    
    // Height and width of a convolution kernel.
    int kernelHeight, kernelWidth;
    /** Kernel dimension - number of unique weight parameters. */
    int kernelDim;
    
    // TODO implement stride and padding
    int strideHeight = 1, strideWidth = 1;
    int padHeight = 0, padWidth = 0;
};

#endif	/* CONVOLUTIONALLAYER_H */

