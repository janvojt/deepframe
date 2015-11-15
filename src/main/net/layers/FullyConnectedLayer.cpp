/* 
 * File:   FullyConnectedLayer.cpp
 * Author: janvojt
 * 
 * Created on May 17, 2015, 12:55 AM
 */

#include "FullyConnectedLayer.h"

#include <algorithm>
#include <sstream>
#include "../../common.h"
#include "../LayerFactory.h"
#include "../../util/cudaHelpers.h"
//#include "../../util/cudaDebugHelpers.h"
#include "../../util/cpuDebugHelpers.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FullyConnectedLayer::FullyConnectedLayer() {
}

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedLayer& orig) {
}

FullyConnectedLayer::~FullyConnectedLayer() {
}

void FullyConnectedLayer::setup(string confString) {
    
    processConfString(confString);
    
    if (previousLayer != NULL) {
        // this is not the input layer
        int prevOutputsCount = previousLayer->getOutputsCount();
        this->weightsCount = prevOutputsCount * conf.outputSize;
        thisOutputDerivatives = new data_t[prevOutputsCount];
        if (conf.useBias) {
            this->weightsCount += conf.outputSize;
        }
    } else {
        this->weightsCount = 0;
    }
    outputsCount = conf.outputSize;
    LOG()->debug("Fully connected layer size is %d neurons.", outputsCount);
}

void FullyConnectedLayer::forwardCpu() {
    
    int inputSize = this->previousLayer->getOutputsCount();
    data_t *inputPtr = this->previousLayer->getOutputs();
    
    // Clear output neurons
    std::fill_n(outputs, conf.outputSize, 0);
    
    data_t *weightPtr = this->weights;
    for (int i = 0; i<inputSize; i++) {
        for (int j = 0; j<conf.outputSize; j++) {
            outputs[j] += inputPtr[i] * *weightPtr;
            weightPtr++;
        }
    }
    
    // Apply bias
    if (conf.useBias) {
        for (int i = 0; i<conf.outputSize; i++) {
            outputs[i] += *weightPtr;
//            LOG()->debug("Input %d after applying bias: %f.", i, outputPtr[i]);
            weightPtr++;
        }
    }
    
    // Run through activation function
    netConf->activationFnc(outputs, outputs, conf.outputSize);
//    dumpHostArray('O', outputPtr, outputsCount);
}

void FullyConnectedLayer::forwardGpu() {
    
    int inputSize = this->previousLayer->getOutputsCount();
    data_t *inputs = this->previousLayer->getOutputs();

//    dumpDeviceArray('I', inputPtr, inputSize);
//    dumpDeviceArray('i', weights, inputSize * outputsCount);
    
    //note cuBLAS is column primary!
    //need to transpose the order
    k_gemm(this->cublasHandle,
            CblasTrans, CblasNoTrans,
            /*n*/outputsCount,/*m*/ 1,/*k*/ inputSize,
            (data_t) 1., weights,
            inputs, (data_t) 0., outputs);

    if (conf.useBias) {
        k_sumVectors(outputs, weights + inputSize * outputsCount, outputsCount);
    }
    
    k_computeSigmoid(outputs, outputsCount);
//    dumpDeviceArray('O', outputs, outputsCount);
}


void FullyConnectedLayer::backwardCpu() {
    // Initialize helper variables
    int prevOutputsCount = previousLayer->getOutputsCount();
    data_t* prevOutputs = previousLayer->getOutputs();
    
    // COMPUTE TOTAL DERIVATIVES for weights between previous and this layer
    for (int i = 0; i<prevOutputsCount; i++) {
        for (int j = 0; j<outputsCount; j++) {
            weightDiffs[i*outputsCount+j] = -lr * outputDiffs[j] * prevOutputs[i];
        }
    }
//    dumpHostArray('w', weightDiffs, outputsCount * prevOutputsCount);

    // COMPUTE BIAS DERIVATIVES for this layer
    if (conf.useBias) {
        data_t *biasDiff = weightDiffs + weightsCount - outputsCount;
        for (int i = 0; i<outputsCount; i++) {
            biasDiff[i] = -lr * outputDiffs[i];
        }
//        dumpHostArray('b', biasDiff, outputsCount);
    }
        
    // COMPUTE LOCAL GRADIENTS for previous layer
    if (!previousLayer->isFirst()) {
    
        // compute derivatives of neuron inputs for this layer
        void (*daf) (data_t*,data_t*,int) = netConf->dActivationFnc;
        daf(prevOutputs, thisOutputDerivatives, prevOutputsCount);

        // compute local gradients for previous layer
        data_t* prevOutputDiffs = previousLayer->getOutputDiffs();
        for (int i = 0; i<prevOutputsCount; i++) {
            data_t gradientSum = 0;
            for (int j = 0; j<outputsCount; j++) {
                gradientSum += outputDiffs[j] * weights[i * outputsCount + j];
            }
            prevOutputDiffs[i] = gradientSum * thisOutputDerivatives[i];
//            LOG()->debug("Local gradient for neuron [%d, %d] : %f.", l, i, thisLocalGradient[i]);
        }
//        dumpHostArray('l', outputDiffs, outputsCount);
    }
}

void FullyConnectedLayer::backwardLastCpu(data_t* expectedOutput) {
    
//    LOG()->debug("Backpropagating (%f, %f) -> (%f).", *(outputs-4), *(outputs-3), *expectedOutput);
    
    void (*daf) (data_t*,data_t*,int) = netConf->dActivationFnc;
    
    // compute local gradients
    daf(outputs, thisOutputDerivatives, outputsCount);
    for (int i = 0; i<outputsCount; i++) {
        outputDiffs[i] = (outputs[i] - expectedOutput[i]) * thisOutputDerivatives[i];
    }
//    dumpHostArray('o', outputDiffs, outputsCount);
}

void FullyConnectedLayer::backwardGpu() {
    // Initialize helper variables
    int prevOutputsCount = previousLayer->getOutputsCount();
    data_t* prevOutputs = previousLayer->getOutputs();
    
    // COMPUTE TOTAL DERIVATIVES for weights between previous and this layer
    k_computeTotalDerivative(prevOutputsCount, outputsCount,
            lr, prevOutputs, outputDiffs,
            weightDiffs);
//    dumpDeviceArray('w', nextWeightDiffs, outputsCount * nextNeurons);
        
    // COMPUTE BIAS DERIVATIVES for layer l
    if (conf.useBias) {
        k_computeBiasDerivative(
                lr, outputDiffs,
                weightDiffs + (prevOutputsCount * outputsCount), // nextBiasDiffs
                outputsCount);
//        dumpDeviceArray('b', weightDiffs + (prevOutputsCount * outputsCount), outputsCount);
    }
    
        
    // COMPUTE LOCAL GRADIENTS for previous layer
    if (!previousLayer->isFirst()) {

        // COMPUTE LOCAL GRADIENTS for this layer
        data_t* prevOutputDiffs = previousLayer->getOutputDiffs();
        k_computeHiddenLocalGradient(
                prevOutputsCount, outputsCount,
                prevOutputs, weights,
                prevOutputDiffs, outputDiffs);
//        dumpDeviceArray('l', outputDiffs, outputsCount + nextNeurons);
    }
}

void FullyConnectedLayer::backwardLastGpu(data_t* expectedOutput) {
    int memSize = outputsCount * sizeof(data_t);
    data_t *dExpOutput;
    checkCudaErrors(cudaMalloc(&dExpOutput, memSize));
    checkCudaErrors(cudaMemcpy(dExpOutput, expectedOutput, memSize, cudaMemcpyHostToDevice));
    k_computeOutputLocalGradient(outputs, dExpOutput, outputDiffs, outputsCount);
    cudaFree(dExpOutput);
//    dumpDeviceArray('L', outputDiffs, outputsCount);
}

void FullyConnectedLayer::processConfString(string confString) {
    // dummy variable for delimiters
    char sep;
    istringstream iss (confString);
    
    if (!(iss >> conf.outputSize)) {
        LOG()->error("Could not read output size for FullyConnected layer.");
    }
    
    iss >> sep;
    
    if (!(iss >> lr)) {
        LOG()->warn("Could not read learning rate for FullyConnected layer from configuration. Using default of 0.3.");
        lr = .3;
    }
    
    iss >> sep;
    
    if (!(iss >> boolalpha >> conf.useBias)) {
        LOG()->warn("Could not read bias for FullyConnected layer from configuration. Not using bias...");
        conf.useBias = false;
    }
}

static LayerRegister<FullyConnectedLayer> reg("FullyConnected");