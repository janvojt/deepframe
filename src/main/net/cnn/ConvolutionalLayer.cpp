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
void ConvolutionalLayer<dType>::setup(Layer<dType> *previousLayer, ConvolutionalConfig conf) {
    this->conf = conf;
    this->previousLayer = previousLayer;
    this->previousLayer->setNextLayer(this);
}

template<typename dType>
void ConvolutionalLayer<dType>::forward() {
    
    dType *inputPtr = this->previousLayer->getInputs();
    dType *outputPtr = this->getInputs();

    // TODO precompute at setup
    int noFeatures = conf.inputFeatures * conf.featureMultiplier;
    int featureWidth = conf.inputWidth - conf.windowSize + 1;
    int featureHeight = conf.inputHeight - conf.windowSize + 1;
    
    // clear output
    std::fill_n(outputPtr, this->getOutputCount(), 0);

    // loop through destination neuron
    for (int f = 0; f < noFeatures; f++) { // destination feature index
        int featureIdx = f * featureHeight * featureWidth;
        
        for (int i = 0; i < featureHeight; i++) { // row index
            int rowIdx = featureIdx + i * featureWidth;
            
            for (int j = 0; j < featureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                
                // loop through source neurons
                for (int pf = 0; pf < conf.inputFeatures; pf++) { // source feature index
                    for (int k = 0; k < conf.windowSize; k++) { // row index
                        for (int l = 0; l < conf.windowSize; l++) { // column index
                            
                            int srcNeuronIdx = pf * conf.windowSize * conf.windowSize
                                                + (k + i) * conf.windowSize + (l + j);
                            
                            int weightIdx = pf * noFeatures * conf.windowSize * conf.windowSize
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
int ConvolutionalLayer<dType>::getOutputCount() {
    return conf.inputFeatures * conf.featureMultiplier
            * (conf.inputWidth - conf.windowSize + 1)
            * (conf.inputHeight - conf.windowSize + 1);
}

template<typename dType>
int ConvolutionalLayer<dType>::getWeightCount() {
    return conf.inputFeatures * conf.featureMultiplier
            * conf.windowSize * conf.windowSize;
}


INSTANTIATE_DATA_CLASS(ConvolutionalLayer);