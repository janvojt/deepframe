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
    featuresCount = conf.featuresCount;

    inputFeatureWidth = subsamplingLayer->getFeatureWidth();
    inputFeatureHeight = subsamplingLayer->getFeatureHeight();
    
    kernelWidth = conf.windowWidth;
    kernelHeight = conf.windowHeight;

    featureWidth = inputFeatureWidth - kernelWidth + 1;
    featureHeight = inputFeatureHeight - kernelHeight + 1;
    
    outputsCount = featuresCount
            * featureWidth * featureHeight;

    genuineWeightsCount = featuresCount
            * kernelWidth * kernelHeight;
    
    if (conf.useBias) {
        weightsCount = genuineWeightsCount + outputsCount;
    } else {
        weightsCount = genuineWeightsCount;
    }
    
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
                    
                    for (int k = 0; k < kernelHeight; k++) { // row index
                        for (int l = 0; l < kernelWidth; l++) { // column index
                            
                            int srcNeuronIdx = srcFeatureIdx + (k + i) * inputFeatureWidth + (l + j);
                            
                            int weightIdx = f * kernelWidth * kernelHeight
                                            + k * kernelWidth + l;
                            
                            outputs[dstNeuronIdx] += inputPtr[srcNeuronIdx] * weights[weightIdx];
                            
                        }
                    }
                }
                
            } // end loop through destination neuron
        }
    }
    
    // apply bias in a separate loop (performs better)
    if (conf.useBias) {

        // loop through destination neuron
        for (int f = 0; f < featuresCount; f++) { // destination feature index
            int featureIdx = f * featureHeight * featureWidth;

            for (int i = 0; i < featureHeight; i++) { // row index
                int rowIdx = featureIdx + i * featureWidth;

                for (int j = 0; j < featureWidth; j++) { // column index
                    
                    int dstNeuronIdx = rowIdx + j;
                    int weightIdx = genuineWeightsCount + dstNeuronIdx;
                    
                    outputs[dstNeuronIdx] += weights[weightIdx];
                }
            }
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
                    
                    for (int k = 0; k < kernelHeight; k++) { // row index
                        for (int l = 0; l < kernelWidth; l++) { // column index
                            
                            int srcNeuronIdx = srcFeatureIdx + (k + i) * inputFeatureWidth + (l + j);
                            
                            int weightIdx = f * kernelWidth * kernelHeight
                                            + k * kernelWidth + l;
                            
                            weightDiffs[weightIdx] += lr * outputDiffs[dstNeuronIdx] * inputs[srcNeuronIdx];
                            inputDiffs[srcNeuronIdx] += lr * outputDiffs[dstNeuronIdx] * weights[weightIdx];
                        }
                    }
                }
                
            } // end loop through destination neuron
        }
    }
    
    // apply bias in a separate loop (performs better)
    if (conf.useBias) {

        // loop through destination neuron
        for (int f = 0; f < featuresCount; f++) { // destination feature index
            int featureIdx = f * featureHeight * featureWidth;

            for (int i = 0; i < featureHeight; i++) { // row index
                int rowIdx = featureIdx + i * featureWidth;

                for (int j = 0; j < featureWidth; j++) { // column index
                    
                    int dstNeuronIdx = rowIdx + j;
                    int weightIdx = genuineWeightsCount + dstNeuronIdx;
                    
                    weightDiffs[weightIdx] += lr * outputDiffs[dstNeuronIdx];
                }
            }
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
        const data_t* outputDiffs, data_t* weightDiffs) {
    
    k_conv_im2col(input, colBuffer);
    
    k_gemm(cublasHandle, CblasNoTrans, CblasTrans, featuresCount,
            kernelDim, featureSize,
            lr, outputDiffs, colBuffer,
            (data_t) 0., weightDiffs);
    
    if (conf.useBias) {
        k_axpy(cublasHandle, outputsCount, lr, outputDiffs, 1, weightDiffs + genuineWeightsCount, 1);
    }
}

/**
 * I = O * W + I
 * 
 * @param output
 * @param weights
 * @param input
 */
void ConvolutionalLayer::k_backwardGemm(const data_t* outputDiffs,
        const data_t* weights, data_t* inputDiffs) {
    
    k_gemm(cublasHandle, CblasTrans, CblasNoTrans, kernelDim,
            featureSize, featuresCount,
            lr, weights, outputDiffs,
            (data_t) 0., colBuffer);

    k_conv_col2im(colBuffer, inputDiffs);
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
    if (!(iss >> conf.windowWidth)) {
        LOG()->error("Could not read window width for Convolutional layer from configuration.");
    }
    
    iss >> sep;
    if (!(iss >> conf.windowHeight)) {
        LOG()->error("Could not read window height for Convolutional layer from configuration.");
    }
    
    iss >> sep;
    if (!(iss >> conf.featuresCount)) {
        LOG()->error("Could not read feature multiplier for Convolutional layer from configuration.");
    }
    
    iss >> sep;
    
    if (!(iss >> lr)) {
        LOG()->warn("Could not read learning rate for Convolutional layer from configuration. Using default of 1.");
        lr = 1;
    }
    
    iss >> sep;
    
    if (!(iss >> boolalpha >> conf.useBias)) {
        LOG()->warn("Could not read bias configuration for Convolutional layer. Not using bias...");
        conf.useBias = false;
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
