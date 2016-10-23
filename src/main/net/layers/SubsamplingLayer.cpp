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
#include <float.h>
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
        inputFeatureWidth = conf.windowWidth;
        inputFeatureHeight = conf.windowHeight;
        featureWidth = conf.windowWidth;
        featureHeight = conf.windowHeight;
        featuresCount = 1;
        outputsCount = conf.windowWidth * conf.windowHeight;
        weightsCount = 0;
    } else {
        ConvolutionalLayer *convLayer = (ConvolutionalLayer*) previousLayer;
        featuresCount = convLayer->getOutputFeatures();
        inputFeatureWidth = convLayer->getOutputWidth();
        inputFeatureHeight = convLayer->getOutputHeight();

        featureWidth = (inputFeatureWidth + conf.windowWidth - 1)  / conf.windowWidth; // round up
        featureHeight = (inputFeatureHeight + conf.windowHeight - 1) / conf.windowHeight; // round up
        outputsCount = featureWidth * featureHeight * featuresCount;

        // subsampling layer does not need any weights
        // but uses a trainable parameter for each feature map
        // and optionally bias for each feature map
//        weightsCount = conf.useBias ? featuresCount*2 : featuresCount;
        weightsCount = featuresCount;
        
        maxIndices = new int[outputsCount];
        if (netConf->getUseGpu()) {
            checkCudaErrors(cudaMalloc(&d_maxIndices, outputsCount * sizeof(int)));
        }
        
        strideHeight = conf.windowHeight;
        strideWidth = conf.windowWidth;
    }
}

void SubsamplingLayer::forwardCpu() {
    
    data_t *inputs = previousLayer->getOutputs();
    int wfeatureWidth = inputFeatureWidth / conf.windowWidth;
    int wfeatureHeight = inputFeatureHeight / conf.windowHeight;
    
    // clear output
    std::fill_n(outputs, outputsCount, 0);
    
    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) {
        int dstFeatureIdx = f * wfeatureWidth * wfeatureHeight;
        int srcFeatureIdx = f * inputFeatureWidth * inputFeatureHeight;
        
        for (int i = 0; i < wfeatureHeight; i++) { // row index
            int rowIdx = dstFeatureIdx + i * wfeatureWidth;
            
            for (int j = 0; j < wfeatureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                // set maximum to the lowest value possible
                outputs[dstNeuronIdx] = -FLT_MAX;
                
                // loop through source neurons
                for (int k = 0; k < conf.windowHeight; k++) { // row index
                    for (int l = 0; l < conf.windowWidth; l++) { // column index

                        int srcNeuronIdx =  srcFeatureIdx + (k + i*conf.windowHeight) * inputFeatureWidth + (l + j*conf.windowWidth);
                        if (inputs[srcNeuronIdx] > outputs[dstNeuronIdx]) {
                            outputs[dstNeuronIdx] = inputs[srcNeuronIdx];
                            maxIndices[dstNeuronIdx] = srcNeuronIdx;
                        }
                    }
                }
            } // end loop through destination neuron
        }
    }

//    netConf->activationFnc(outputs, outputs, outputsCount);
    
    // TODO if (conf.inputWidth % conf.windowWidth > 0)
    // TODO if (conf.inputHeight % conf.windowHeight > 0)
}


void SubsamplingLayer::forwardGpu() {
    
    const data_t* inputs = previousLayer->getOutputs();
    k_MaxPoolForward(
            outputsCount, inputs, featuresCount,
            inputFeatureHeight, inputFeatureWidth, featureHeight, featureWidth,
            conf.windowHeight, conf.windowWidth, strideHeight, strideWidth,
            padHeight, padWidth, outputs, d_maxIndices);
}


void SubsamplingLayer::backwardCpu() {
    
    data_t *inputDiffs = previousLayer->getOutputDiffs();
    int wfeatureWidth = inputFeatureWidth / conf.windowWidth;
    int wfeatureHeight = inputFeatureHeight / conf.windowHeight;
    
    // clear output
    std::fill_n(inputDiffs, previousLayer->getOutputsCount(), 0);
    
    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) {
        int dstFeatureIdx = f * wfeatureWidth * wfeatureHeight;
        
        for (int i = 0; i < wfeatureHeight; i++) { // row index
            int rowIdx = dstFeatureIdx + i * wfeatureWidth;
            
            for (int j = 0; j < wfeatureWidth; j++) { // column index

                int dstNeuronIdx = rowIdx + j;
                inputDiffs[maxIndices[dstNeuronIdx]] = lr * outputDiffs[dstNeuronIdx];

            } // end loop through destination neuron
        }
    }
    
    // TODO if (conf.inputWidth % conf.windowWidth > 0)
    // TODO if (conf.inputHeight % conf.windowHeight > 0)
}

void SubsamplingLayer::backwardGpu() {

    data_t* inputDiffs = previousLayer->getOutputDiffs();
    int inputsCount = previousLayer->getOutputsCount();
    cudaMemset(inputDiffs, 0, inputsCount * sizeof(data_t));

    k_MaxPoolBackward(
            inputsCount, outputDiffs, d_maxIndices, featuresCount,
            inputFeatureHeight, inputFeatureWidth, featureHeight, featureWidth,
            conf.windowHeight, conf.windowWidth, strideHeight, strideWidth, padHeight, padWidth,
            inputDiffs);
    
    if (lr != 1) {
        k_scal(cublasHandle, inputsCount, lr, inputDiffs, 1);
    }
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
    
    if (!(iss >> lr)) {
        LOG()->warn("Could not read learning rate for Subsampling layer from configuration. Using default of 1.");
        lr = 1;
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