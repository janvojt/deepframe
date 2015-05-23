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
void SubsamplingLayer<dType>::setup(Layer<dType>* previousLayer, SubsamplingConfig<dType> conf) {
    this->conf = conf;
    if (previousLayer != NULL) {
        // this is not the input layer
        this->previousLayer = previousLayer;
        previousLayer->setNextLayer(this);
    }
}

template<typename dType>
void SubsamplingLayer<dType>::forward() {
    
    dType *inputPtr = this->previousLayer->getInputs();
    dType *outputPtr = this->getInputs();
    int outputCount = this->getOutputCount();
    
    int featureWidth = conf.inputWidth / conf.windowWidth;
    int featureHeight = conf.inputHeight / conf.windowHeight;
    
    // clear output
    std::fill_n(outputPtr, outputCount, 0);
    
    // loop through destination neuron
    for (int f = 0; f < conf.inputFeatures; f++) {
        int dstFeatureIdx = f * featureWidth * featureHeight;
        int srcFeatureIdx = f * conf.inputWidth * conf.inputHeight;
        
        for (int i = 0; i < featureHeight; i++) { // row index
            int rowIdx = dstFeatureIdx + i * featureWidth;
            
            for (int j = 0; j < featureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                int max = -1;
                
                // loop through source neurons
                for (int k = 0; k < conf.inputHeight; k++) { // row index
                    for (int l = 0; l < conf.inputWidth; l++) { // column index

                        int srcNeuronIdx =  srcFeatureIdx + (k+i) * conf.inputWidth + (l+j);

                        max = (inputPtr[srcNeuronIdx] > max) ? inputPtr[srcNeuronIdx] : max;
                    }
                }
                
                outputPtr[dstNeuronIdx] += max * this->weights[f] + this->weights[conf.inputFeatures + f];
                
            } // end loop through destination neuron
        }
    }

    conf.activationFnc(outputPtr, outputPtr, outputCount);
    
    // TODO if (conf.inputWidth % conf.windowWidth > 0)
    // TODO if (conf.inputHeight % conf.windowHeight > 0)
}


template<typename dType>
int SubsamplingLayer<dType>::getWeightCount() {
    // subsampling layer does not need any weights
    // but uses a trainable parameter for each feature map
    // and optionally bias for each feature map
    return conf.useBias ? conf.inputFeatures*2 : conf.inputFeatures;
}

template<typename dType>
int SubsamplingLayer<dType>::getOutputCount() {
    int featureWidth = (conf.inputWidth + conf.windowWidth - 1)  / conf.windowWidth; // round up
    int featureHeight = (conf.inputHeight + conf.windowHeight - 1) / conf.windowHeight; // round up
    return featureWidth * featureHeight * conf.inputFeatures;
}

INSTANTIATE_DATA_CLASS(SubsamplingLayer);