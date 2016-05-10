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

// we need to declare Convolutional layer here, because it is used
// by the Subsampling layer header.
class ConvolutionalLayer;

/** Holds configuration parameters for this subsampling layer. */
struct SubsamplingConfig {
    int windowWidth;
    int windowHeight;
    bool useBias;
};

/**
 * Represents the subsampling layer as a building block of Convolutional Neural Network.
 */
class SubsamplingLayer : public Layer {
public:
    SubsamplingLayer();
    SubsamplingLayer(const SubsamplingLayer& orig);
    virtual ~SubsamplingLayer();

    /**
     * Performs the forward pass during training and testing on CPU.
     */
    void forwardCpu();
    /**
     * Performs the forward pass during training and testing on GPU.
     */
    void forwardGpu();
    
    /**
     * Performs the backward pass during training on CPU.
     */
    void backwardCpu();
    /**
     * Performs the backward pass during training on GPU.
     */
    void backwardGpu();
    
    /**
     * Not implemented for Subsampling layer, as it cannot be the last one.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastCpu(data_t* expectedOutput);
    /**
     * Not implemented for Subsampling layer, as it cannot be the last one.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastGpu(data_t* expectedOutput);
    
    /**
     * @return the configuration of this layer
     */
    SubsamplingConfig getConfig();
    
    /**
     * @return number of features in this layer
     */
    int getFeaturesCount();
    
    /**
     * @return width of the feature maps in this layer
     */
    int getFeatureWidth();
    
    /**
     * @return height of the feature maps in this layer
     */
    int getFeatureHeight();
    
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
    
    /** Configuration for this layer. */
    SubsamplingConfig conf;
    
    /** Number of feature in this layer. */
    int featuresCount;
    
    /** Width of features in this */
    int featureWidth;
    
    /** Height of feature maps in this layer. */
    int featureHeight;
    
    /** Width of feature maps in preceding layer. */
    int inputFeatureWidth;
    
    /** Height of feature maps in preceding layer. */
    int inputFeatureHeight;
    
    /** Device memory for storing indices of activated neurons in pooling. */
    int *d_maxIndices = NULL;
    
    /** Host memory for storing indices of activated neurons in pooling. */
    int *maxIndices = NULL;
    
    // TODO implement stride and padding
    int strideHeight = 1, strideWidth = 1;
    int padHeight = 0, padWidth = 0;
};

#endif	/* SUBSAMPLINGLAYER_H */

