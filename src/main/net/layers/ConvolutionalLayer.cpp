/* 
 * File:   ConvolutionalLayer.cpp
 * Author: janvojt
 * 
 * Created on May 16, 2015, 11:38 PM
 */

#include "ConvolutionalLayer.h"

#include <algorithm>
#include <sstream>
#include "../../common.h"
#include "../LayerFactory.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

ConvolutionalLayer::ConvolutionalLayer() {
}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayer& orig) {
}

ConvolutionalLayer::~ConvolutionalLayer() {
}

void ConvolutionalLayer::setup(string confString) {
    
    if (previousLayer == NULL) {
        LOG()->error("Convolutional layer is not supported as an input layer. Use subsampling layer instead.");
        return;
    }
    
    processConfString(confString);
    
    SubsamplingLayer *subsamplingLayer = (SubsamplingLayer*) previousLayer;
    inputFeatures = subsamplingLayer->getFeaturesCount();
    featuresCount = inputFeatures * conf.featureMultiplier;
    featureWidth = subsamplingLayer->getFeatureWidth() - conf.windowSize + 1;
    featureHeight = subsamplingLayer->getFeatureHeight() - conf.windowSize + 1;
    
    this->outputsCount = featuresCount
            * featureWidth * featureHeight;
    
    this->weightsCount = featuresCount
            * conf.windowSize * conf.windowSize;
}

void ConvolutionalLayer::forwardCpu() {
    
    data_t *inputPtr = this->previousLayer->getOutputs();
    data_t *outputPtr = this->getOutputs();

    // clear output
    std::fill_n(outputPtr, this->outputsCount, 0);

    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) { // destination feature index
        int featureIdx = f * featureHeight * featureWidth;
        
        for (int i = 0; i < featureHeight; i++) { // row index
            int rowIdx = featureIdx + i * featureWidth;
            
            for (int j = 0; j < featureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                
                // loop through source neurons
                for (int pf = 0; pf < inputFeatures; pf++) { // source feature index
                    for (int k = 0; k < conf.windowSize; k++) { // row index
                        for (int l = 0; l < conf.windowSize; l++) { // column index
                            
                            int srcNeuronIdx = pf * conf.windowSize * conf.windowSize
                                                + (k + i) * conf.windowSize + (l + j);
                            
                            int weightIdx = pf * featuresCount * conf.windowSize * conf.windowSize
                                            + k * conf.windowSize + l;
                            
                            outputPtr[dstNeuronIdx] += inputPtr[srcNeuronIdx] * this->weights[weightIdx];
                            
                        }
                    }
                }
                
            } // end loop through destination neuron
        }
    }
}


void ConvolutionalLayer::forwardGpu() {
    //TODO
}


void ConvolutionalLayer::backwardCpu() {
    //TODO
}


void ConvolutionalLayer::backwardGpu() {
    //TODO
}

void ConvolutionalLayer::backwardLastCpu(data_t* expectedOutput) {
    LOG()->error("Backpropagation based on expected output is not implemented in Convolutional layer. This error happens when Convolutional layer is the last network layer.");
}

void ConvolutionalLayer::backwardLastGpu(data_t* expectedOutput) {
    LOG()->error("Backpropagation based on expected output is not implemented in Convolutional layer. This error happens when Convolutional layer is the last network layer.");
}


void ConvolutionalLayer::processConfString(string confString) {
    // dummy variable for delimiters
    char sep;
    
    istringstream iss (confString);
    if (!(iss >> conf.windowSize)) {
        LOG()->error("Could not read window size for Convolutional layer from configuration.");
    }
    iss >> sep;
    if (!(iss >> conf.featureMultiplier)) {
        LOG()->error("Could not read feature multiplier for Convolutional layer from configuration.");
    }
}

ConvolutionalConfig ConvolutionalLayer::getConfig() {
    return conf;
}

int ConvolutionalLayer::getOutputFeatures() {
    return this->featuresCount;
}

int ConvolutionalLayer::getOutputWidth() {
    return this->featureWidth;
}

int ConvolutionalLayer::getOutputHeight() {
    return this->featureHeight;
}

static LayerRegister<ConvolutionalLayer> reg("Convolution");
