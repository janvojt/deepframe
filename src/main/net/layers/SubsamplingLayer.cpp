/* 
 * File:   SubsamplingLayer.cpp
 * Author: janvojt
 * 
 * Created on May 16, 2015, 11:38 PM
 */

#include "SubsamplingLayer.h"

#include <cstdlib>
#include <algorithm>
#include "../../common.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

SubsamplingLayer::SubsamplingLayer() {
}

SubsamplingLayer::SubsamplingLayer(const SubsamplingLayer& orig) {
}

SubsamplingLayer::~SubsamplingLayer() {
}

void SubsamplingLayer::setup(ConvolutionalLayer* previousLayer, SubsamplingConfig conf) {

    this->conf = conf;
    this->previousLayer = previousLayer;
    
    if (previousLayer == NULL) {
        // this is an input layer
        inputWidth = conf.windowWidth;
        inputHeight = conf.windowHeight;
        featureWidth = conf.windowWidth;
        featureHeight = conf.windowHeight;
        featuresCount = 1;
        this->inputsCount = conf.windowWidth * conf.windowHeight;
        this->weightsCount = 0;
    } else {
        previousLayer->setNextLayer(this);

        featuresCount = previousLayer->getOutputFeatures();
        inputWidth = previousLayer->getOutputWidth();
        inputHeight = previousLayer->getOutputHeight();

        featureWidth = (inputWidth + conf.windowWidth - 1)  / conf.windowWidth; // round up
        featureHeight = (inputHeight + conf.windowHeight - 1) / conf.windowHeight; // round up
        this->inputsCount = featureWidth * featureHeight * featuresCount;

        // subsampling layer does not need any weights
        // but uses a trainable parameter for each feature map
        // and optionally bias for each feature map
        this->weightsCount = conf.useBias ? featuresCount*2 : featuresCount;
    }
}

void SubsamplingLayer::forwardCpu() {
    
    data_t *inputPtr = this->previousLayer->getInputs();
    data_t *outputPtr = this->getInputs();
    int outputCount = this->inputsCount;
    
    int wfeatureWidth = inputWidth / conf.windowWidth;
    int wfeatureHeight = inputHeight / conf.windowHeight;
    
    // clear output
    std::fill_n(outputPtr, outputCount, 0);
    
    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) {
        int dstFeatureIdx = f * wfeatureWidth * wfeatureHeight;
        int srcFeatureIdx = f * inputWidth * inputHeight;
        
        for (int i = 0; i < wfeatureHeight; i++) { // row index
            int rowIdx = dstFeatureIdx + i * wfeatureWidth;
            
            for (int j = 0; j < wfeatureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                int max = -1;
                
                // loop through source neurons
                for (int k = 0; k < inputHeight; k++) { // row index
                    for (int l = 0; l < inputWidth; l++) { // column index

                        int srcNeuronIdx =  srcFeatureIdx + (k+i) * inputWidth + (l+j);

                        max = (inputPtr[srcNeuronIdx] > max) ? inputPtr[srcNeuronIdx] : max;
                    }
                }
                
                outputPtr[dstNeuronIdx] += max * this->weights[f] + this->weights[featuresCount + f];
                
            } // end loop through destination neuron
        }
    }

    conf.activationFnc(outputPtr, outputPtr, outputCount);
    
    // TODO if (conf.inputWidth % conf.windowWidth > 0)
    // TODO if (conf.inputHeight % conf.windowHeight > 0)
}


void SubsamplingLayer::forwardGpu() {
    //TODO
}


void SubsamplingLayer::backwardCpu() {
    //TODO
}


void SubsamplingLayer::backwardGpu() {
    //TODO
}


SubsamplingConfig SubsamplingLayer::getConfig() {
    return this->conf;
}

int SubsamplingLayer::getFeatureWidth() {
    return this->featureWidth;
}

int SubsamplingLayer::getFeatureHeight() {
    return this->featureHeight;
}

int SubsamplingLayer::getFeaturesCount() {
    return this->featuresCount;
}
