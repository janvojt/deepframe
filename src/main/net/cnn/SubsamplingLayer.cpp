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

template <typename dType>
SubsamplingLayer<dType>::SubsamplingLayer() {
}

template <typename dType>
SubsamplingLayer<dType>::SubsamplingLayer(const SubsamplingLayer& orig) {
}

template <typename dType>
SubsamplingLayer<dType>::~SubsamplingLayer() {
}

template<typename dType>
void SubsamplingLayer<dType>::setup(ConvolutionalLayer<dType>* previousLayer, SubsamplingConfig<dType> conf) {

    this->conf = conf;
    this->previousLayer = previousLayer;
    previousLayer->setNextLayer(this);
    
    ConvolutionalConfig pConf = previousLayer->getConfig();
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

template<typename dType>
void SubsamplingLayer<dType>::setupAsInput(int outputWidth, int outputHeight) {
    this->inputWidth = outputWidth;
    this->inputHeight = outputHeight;
    this->featureWidth = outputWidth;
    this->featureHeight = outputHeight;
    this->featuresCount = 1;
    this->inputsCount = outputWidth * outputHeight;
    this->weightsCount = 0;
}

template<typename dType>
void SubsamplingLayer<dType>::forward() {
    
    dType *inputPtr = this->previousLayer->getInputs();
    dType *outputPtr = this->getInputs();
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

template<typename dType>
SubsamplingConfig<dType> SubsamplingLayer<dType>::getConfig() {
    return this->conf;
}

template<typename dType>
int SubsamplingLayer<dType>::getFeatureWidth() {
    return this->featureWidth;
}

template<typename dType>
int SubsamplingLayer<dType>::getFeatureHeight() {
    return this->featureHeight;
}

template<typename dType>
int SubsamplingLayer<dType>::getFeaturesCount() {
    return this->featuresCount;
}


INSTANTIATE_DATA_CLASS(SubsamplingLayer);