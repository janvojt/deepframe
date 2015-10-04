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
#include "../../util/cudaDebugHelpers.h"

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

    inputFeatureWidth = subsamplingLayer->getFeatureWidth();
    inputFeatureHeight = subsamplingLayer->getFeatureHeight();

    featureWidth = inputFeatureWidth - conf.windowSize + 1;
    featureHeight = inputFeatureHeight - conf.windowSize + 1;
    
    outputsCount = featuresCount
            * featureWidth * featureHeight;
    
    weightsCount = featuresCount
            * conf.windowSize * conf.windowSize;
    
    kernelHeight = conf.windowSize;
    kernelWidth = conf.windowSize;
    kernelDim = inputFeatures * kernelHeight * kernelWidth;
    featureSize = featureWidth * featureHeight; // feature size
    colWidth = featureSize; // width of column buffer
    colHeight = kernelDim; // height of column buffer
    colSize = colWidth * colHeight;
    
    checkCudaErrors(cudaMalloc(&colBuffer, colSize * sizeof(data_t)));
}

void ConvolutionalLayer::forwardCpu() {
    
    data_t *inputPtr = previousLayer->getOutputs();

    // clear output
    std::fill_n(outputs, outputsCount, 0);

    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) { // destination feature index
        int featureIdx = f * featureHeight * featureWidth;
        
        for (int i = 0; i < featureHeight; i++) { // row index
            int rowIdx = featureIdx + i * featureWidth;
            
            for (int j = 0; j < featureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                
                // loop through source neurons
                for (int pf = 0; pf < inputFeatures; pf++) { // source feature index
                    int srcFeatureIdx = pf * inputFeatureWidth * inputFeatureHeight;
                    
                    for (int k = 0; k < conf.windowSize; k++) { // row index
                        for (int l = 0; l < conf.windowSize; l++) { // column index
                            
                            int srcNeuronIdx = srcFeatureIdx + (k + i) * inputFeatureWidth + (l + j);
                            
                            int weightIdx = f * conf.windowSize * conf.windowSize
                                            + k * conf.windowSize + l;
                            
                            outputs[dstNeuronIdx] += inputPtr[srcNeuronIdx] * weights[weightIdx];
                            
                        }
                    }
                }
                
            } // end loop through destination neuron
        }
    }
}

void ConvolutionalLayer::forwardGpu() {

    const data_t* inputs = previousLayer->getOutputs();
    
    k_conv_im2col(inputs, colBuffer);

    k_gemm(cublasHandle, CblasNoTrans, CblasNoTrans,
            featuresCount, featureSize, kernelDim,
            (data_t) 1., weights, colBuffer,
            (data_t) 0., outputs);
}

void ConvolutionalLayer::backwardCpu() {
    
    data_t *inputs = previousLayer->getOutputs();
    data_t *inputDiffs = previousLayer->getOutputDiffs();

    // clear output
    std::fill_n(inputDiffs, previousLayer->getOutputsCount(), 0);
    std::fill_n(weightDiffs, weightsCount, 0);

    // loop through destination neuron
    for (int f = 0; f < featuresCount; f++) { // destination feature index
        int featureIdx = f * featureHeight * featureWidth;
        
        for (int i = 0; i < featureHeight; i++) { // row index
            int rowIdx = featureIdx + i * featureWidth;
            
            for (int j = 0; j < featureWidth; j++) { // column index
                int dstNeuronIdx = rowIdx + j;
                
                // loop through source neurons
                for (int pf = 0; pf < inputFeatures; pf++) { // source feature index
                    int srcFeatureIdx = pf * inputFeatureWidth * inputFeatureHeight;
                    
                    for (int k = 0; k < conf.windowSize; k++) { // row index
                        for (int l = 0; l < conf.windowSize; l++) { // column index
                            
                            int srcNeuronIdx = srcFeatureIdx + (k + i) * inputFeatureWidth + (l + j);
                            
                            int weightIdx = f * conf.windowSize * conf.windowSize
                                            + k * conf.windowSize + l;
                            
                            weightDiffs[weightIdx] += outputDiffs[dstNeuronIdx] * inputs[srcNeuronIdx];
                            inputDiffs[srcNeuronIdx] += outputDiffs[dstNeuronIdx] * weights[weightIdx];
                        }
                    }
                }
                
            } // end loop through destination neuron
        }
    }
}

/**
 * W = I * O + W
 * 
 * @param input
 * @param output
 * @param weights
 */
void ConvolutionalLayer::k_weightGemm(const data_t* input,
        const data_t* output, data_t* weights) {
    
    k_conv_im2col(input, colBuffer);
    
    k_gemm(cublasHandle, CblasNoTrans, CblasTrans, featuresCount,
            kernelDim, featureSize,
            (data_t) 1., output, colBuffer,
            (data_t) 1., weights);
}

/**
 * I = O * W + I
 * 
 * @param output
 * @param weights
 * @param input
 */
void ConvolutionalLayer::k_backwardGemm(const data_t* output,
        const data_t* weights, data_t* input) {
    
    k_gemm(cublasHandle, CblasTrans, CblasNoTrans, kernelDim,
            featureSize, featuresCount,
            (data_t) 1., weights, output,
            (data_t) 0., colBuffer);

    k_conv_col2im(colBuffer, input);
}

void ConvolutionalLayer::backwardGpu() {
    
    const data_t* inputs = previousLayer->getOutputs();
    data_t* inputDiffs = previousLayer->getOutputDiffs();
    
    // gradient w.r.t. weight. Note that we will accumulate diffs.
    k_weightGemm(inputs, outputDiffs, weightDiffs);
    
    // gradient w.r.t. bottom data, if necessary.
    k_backwardGemm(outputDiffs, weights, inputDiffs);
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
