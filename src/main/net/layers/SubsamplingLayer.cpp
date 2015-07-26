/* 
 * File:   SubsamplingLayer.cpp
 * Author: janvojt
 * 
 * Created on May 16, 2015, 11:38 PM
 */

#include "SubsamplingLayer.h"

#include <cstdlib>
#include <algorithm>
#include <sstream>
#include "../../common.h"
#include "../LayerFactory.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

SubsamplingLayer::SubsamplingLayer() {
}

SubsamplingLayer::SubsamplingLayer(const SubsamplingLayer& orig) {
}

SubsamplingLayer::~SubsamplingLayer() {
}

void SubsamplingLayer::setup(string confString) {

    processConfString(confString);
    
    if (previousLayer == NULL) {
        // this is an input layer
        inputWidth = conf.windowWidth;
        inputHeight = conf.windowHeight;
        featureWidth = conf.windowWidth;
        featureHeight = conf.windowHeight;
        featuresCount = 1;
        this->outputsCount = conf.windowWidth * conf.windowHeight;
        this->weightsCount = 0;
    } else {
        ConvolutionalLayer *convLayer = (ConvolutionalLayer*) previousLayer;
        featuresCount = convLayer->getOutputFeatures();
        inputWidth = convLayer->getOutputWidth();
        inputHeight = convLayer->getOutputHeight();

        featureWidth = (inputWidth + conf.windowWidth - 1)  / conf.windowWidth; // round up
        featureHeight = (inputHeight + conf.windowHeight - 1) / conf.windowHeight; // round up
        this->outputsCount = featureWidth * featureHeight * featuresCount;

        // subsampling layer does not need any weights
        // but uses a trainable parameter for each feature map
        // and optionally bias for each feature map
        this->weightsCount = conf.useBias ? featuresCount*2 : featuresCount;
    }
}

void SubsamplingLayer::forwardCpu() {
    
    data_t *inputs = this->previousLayer->getOutputs();
    int wfeatureWidth = inputWidth / conf.windowWidth;
    int wfeatureHeight = inputHeight / conf.windowHeight;
    
    // clear output
    std::fill_n(outputs, outputsCount, 0);
    
    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) {
        int dstFeatureIdx = f * wfeatureWidth * wfeatureHeight;
        int srcFeatureIdx = f * inputWidth * inputHeight;
        
        for (int i = 0; i < wfeatureHeight; i++) { // row index
            int rowIdx = dstFeatureIdx + i * wfeatureWidth;
            
            for (int j = 0; j < wfeatureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                data_t max = -1;
                
                // loop through source neurons
                for (int k = 0; k < inputHeight; k++) { // row index
                    for (int l = 0; l < inputWidth; l++) { // column index

                        int srcNeuronIdx =  srcFeatureIdx + (k+i) * inputWidth + (l+j);

                        max = (inputs[srcNeuronIdx] > max) ? inputs[srcNeuronIdx] : max;
                    }
                }
                
                outputs[dstNeuronIdx] += max; // * this->weights[f] + this->weights[featuresCount + f];
                
            } // end loop through destination neuron
        }
    }

    netConf->activationFnc(outputs, outputs, outputsCount);
    
    // TODO if (conf.inputWidth % conf.windowWidth > 0)
    // TODO if (conf.inputHeight % conf.windowHeight > 0)
}


void SubsamplingLayer::forwardGpu() {
    //TODO
}


void SubsamplingLayer::backwardCpu() {
    
    int inputsCount = previousLayer->getOutputsCount();
    data_t *inputDiffs = previousLayer->getOutputDiffs();
    int wfeatureWidth = inputWidth / conf.windowWidth;
    int wfeatureHeight = inputHeight / conf.windowHeight;
    
    // clear output
    std::fill_n(inputDiffs, inputsCount, 0);
    
    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) {
        int dstFeatureIdx = f * wfeatureWidth * wfeatureHeight;
        int srcFeatureIdx = f * inputWidth * inputHeight;
        
        for (int i = 0; i < wfeatureHeight; i++) { // row index
            int rowIdx = dstFeatureIdx + i * wfeatureWidth;
            
            for (int j = 0; j < wfeatureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                
                // loop through source neurons
                for (int k = 0; k < inputHeight; k++) { // row index
                    for (int l = 0; l < inputWidth; l++) { // column index

                        int srcNeuronIdx =  srcFeatureIdx + (k+i) * inputWidth + (l+j);
                        inputDiffs[srcNeuronIdx] += outputDiffs[dstNeuronIdx];
                    }
                }
            } // end loop through destination neuron
        }
    }
}


void SubsamplingLayer::backwardGpu() {
    //TODO
}

void SubsamplingLayer::backwardLastCpu(data_t* expectedOutput) {
    LOG()->error("Backpropagation based on expected output is not implemented in Subsampling layer. This error happens when Subsampling layer is the last network layer.");
}

void SubsamplingLayer::backwardLastGpu(data_t* expectedOutput) {
    LOG()->error("Backpropagation based on expected output is not implemented in Subsampling layer. This error happens when Subsampling layer is the last network layer.");
}

SubsamplingConfig SubsamplingLayer::getConfig() {
    return this->conf;
}

void SubsamplingLayer::processConfString(string confString) {
    // dummy variable for reading delimiters
    char sep;
    
    istringstream iss (confString);
    if (!(iss >> conf.windowWidth)) {
        LOG()->error("Could not read window width for Subsampling layer from configuration.");
    }
    iss >> sep;
    if (!(iss >> conf.windowHeight)) {
        LOG()->error("Could not read window height for Subsampling layer from configuration.");
    }
    iss >> sep;
    if (!(iss >> boolalpha >> conf.useBias)) {
        LOG()->warn("Could not read bias configuration for Subsampling layer. Not using bias...");
        conf.useBias = false;
    }
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

static LayerRegister<SubsamplingLayer> reg("Subsampling");