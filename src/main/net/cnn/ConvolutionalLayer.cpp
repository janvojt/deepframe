/* 
 * File:   ConvolutionalLayer.cpp
 * Author: janvojt
 * 
 * Created on May 16, 2015, 11:38 PM
 */

#include "ConvolutionalLayer.h"

#include <algorithm>
#include "../../common.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template<typename dType>
ConvolutionalLayer<dType>::ConvolutionalLayer() {
}

template<typename dType>
ConvolutionalLayer<dType>::ConvolutionalLayer(const ConvolutionalLayer& orig) {
}

template<typename dType>
ConvolutionalLayer<dType>::~ConvolutionalLayer() {
}

template<typename dType>
void ConvolutionalLayer<dType>::setup(SubsamplingLayer<dType> *previousLayer, ConvolutionalConfig conf) {
    this->conf = conf;
    this->previousLayer = previousLayer;
    this->previousLayer->setNextLayer(this);
    
    inputFeatures = previousLayer->getFeaturesCount();
    featuresCount = inputFeatures * conf.featureMultiplier;
    featureWidth = previousLayer->getFeatureWidth() - conf.windowSize + 1;
    featureHeight = previousLayer->getFeatureHeight() - conf.windowSize + 1;
    
    this->inputsCount = featuresCount
            * featureWidth * featureHeight;
    
    this->weightsCount = featuresCount
            * conf.windowSize * conf.windowSize;
}

template<typename dType>
void ConvolutionalLayer<dType>::forward() {
    
    dType *inputPtr = this->previousLayer->getInputs();
    dType *outputPtr = this->getInputs();

    // clear output
    std::fill_n(outputPtr, this->inputsCount, 0);

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

template<typename dType>
ConvolutionalConfig ConvolutionalLayer<dType>::getConfig() {
    return conf;
}

template<typename dType>
int ConvolutionalLayer<dType>::getOutputFeatures() {
    return this->featuresCount;
}

template<typename dType>
int ConvolutionalLayer<dType>::getOutputWidth() {
    return this->featureWidth;
}

template<typename dType>
int ConvolutionalLayer<dType>::getOutputHeight() {
    return this->featureHeight;
}


INSTANTIATE_DATA_CLASS(ConvolutionalLayer);