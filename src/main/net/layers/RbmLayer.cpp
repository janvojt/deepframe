/* 
 * File:   RbmLayer.cpp
 * Author: janvojt
 * 
 * Created on November 11, 2015, 8:43 PM
 */

#include "RbmLayer.h"
#include "../../common.h"
#include "../LayerFactory.h"
#include "../../util/cudaHelpers.h"
//#include "../../util/cudaDebugHelpers.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

RbmLayer::RbmLayer() {
}

RbmLayer::RbmLayer(const RbmLayer& orig) {
}

RbmLayer::~RbmLayer() {
    
    if (!isLayerSetup) {
        return;
    }
    
    if (netConf->getUseGpu()) {
        // TODO free memory held by CUDA
    } else {
        delete[] ohPotentials;
    }
}

void RbmLayer::setup(string confString) {
    
    isLayerSetup = true;
    processConfString(confString);
    
    outputsCount = conf.outputSize;
    
    inputSize = previousLayer->getOutputsCount();
    genuineWeightsCount = inputSize * outputsCount;
    
    if (conf.useBias) {
        weightsCount = genuineWeightsCount + inputSize + outputsCount;
    } else {
        weightsCount = genuineWeightsCount;
    }
    
    if (netConf->getUseGpu()) {
        // arrays for storing sampled input and output
        checkCudaErrors(cudaMalloc(&sInputs, inputSize * sizeof(data_t)));
        checkCudaErrors(cudaMalloc(&sOutputs, outputsCount * sizeof(data_t)));

        // arrays for storing original potentials
        checkCudaErrors(cudaMalloc(&ovPotentials, inputSize * sizeof(data_t)));
        checkCudaErrors(cudaMalloc(&ohPotentials, outputsCount * sizeof(data_t)));

        // arrays for storing sampled potentials
        checkCudaErrors(cudaMalloc(&svPotentials, inputSize * sizeof(data_t)));
        checkCudaErrors(cudaMalloc(&shPotentials, outputsCount * sizeof(data_t)));

        int memCount = outputsCount > inputSize ? outputsCount : inputSize;
        checkCudaErrors(cudaMalloc(&randomData, memCount * sizeof(data_t)));
    } else {
        ohPotentials = new data_t[outputsCount];
    }
}

void RbmLayer::forwardCpu() {
    propagateForwardCpu(previousLayer->getOutputs(), ohPotentials, outputs);
}

void RbmLayer::forwardGpu() {
    propagateForwardGpu(previousLayer->getOutputs(), ohPotentials, outputs);
}

void RbmLayer::backwardCpu() {
    // DO NOTHING
    // There is no need to backpropagate, because the RBM layers are learning
    // only in the pretraining phase. See #pretrain().
}

void RbmLayer::backwardGpu() {
    // DO NOTHING
    // There is no need to backpropagate, because the RBM layers are learning
    // only in the pretraining phase. See #pretrain().
}

void RbmLayer::propagateForwardCpu(data_t* visibles, data_t* potentials, data_t* hiddens) {

    // Clear output neurons
    std::fill_n(potentials, outputsCount, 0);

    for (int i = 0; i<inputSize; i++) {
        for (int j = 0; j<outputsCount; j++) {
            int idx = i + j * inputSize;
            potentials[j] += visibles[i] * weights[idx];
        }
    }

    // Apply bias
    data_t *hbias = weights + inputSize * outputsCount + inputSize;
    if (conf.useBias) {
        for (int i = 0; i<outputsCount; i++) {
            potentials[i] += hbias[i];
        }
    }

    // Run through activation function
    netConf->activationFnc(potentials, hiddens, outputsCount);
//    dumpHostArray('O', outputPtr, outputsCount);
}

void RbmLayer::propagateForwardGpu(data_t* visibles, data_t* potentials, data_t* hiddens) {
    
    data_t *hbias = weights + inputSize * outputsCount + inputSize;
    
    // propagate potentials from visible to hidden neurons
    k_gemm(this->cublasHandle,
            CblasNoTrans, CblasNoTrans,
            /*n*/outputsCount,/*m*/ 1,/*k*/ inputSize,
            (data_t) 1., weights,
            visibles, (data_t) 0., potentials);

    // apply bias
    if (conf.useBias) {
        k_sumVectors(potentials, hbias, outputsCount);
    }
    
    k_computeSigmoid(potentials, hiddens, outputsCount);
}

void RbmLayer::propagateBackwardCpu(data_t* hiddens, data_t* potentials, data_t* visibles) {
    LOG()->error("RBM layer does not implement CPU support in this version.");
}

void RbmLayer::propagateBackwardGpu(data_t* hiddens, data_t* potentials, data_t* visibles) {
    
    // propagate potentials from hidden back to visible neurons
    k_gemm(this->cublasHandle,
            CblasTrans, CblasNoTrans,
            /*n*/inputSize,/*m*/ 1,/*k*/ outputsCount,
            (data_t) 1., weights,
            hiddens, (data_t) 0., potentials);

    // apply bias
    if (conf.useBias) {
        data_t *vbias = weights + inputSize * outputsCount;
        k_sumVectors(potentials, vbias, inputSize);
    }
    
    k_computeSigmoid(potentials, visibles, inputSize);
}

void RbmLayer::pretrainCpu() {
    LOG()->error("Pretraining RBM layer on CPU is not supported in this version. Please pretrain on GPU.");
}

void RbmLayer::pretrainGpu() {
    
    data_t *inputs =  previousLayer->getOutputs();
    
    // single forward run will compute original results
    // needed to compute the differentials
    forwardGpu();

    // sample binary states from potentials
    k_generateUniform(*curandGen, randomData, outputsCount);
    k_uniformToCoinFlip(outputs, randomData, outputsCount);
    
    // Reset the parameters sampled from previous training
    if (!conf.isPersistent || !samplesInitialized) {
        samplesInitialized = true;
        sample_vh_gpu(inputs, shPotentials, sOutputs);
    }
    
    // perform CD-k
    gibbs_hvh(conf.gibbsSteps);

    // COMPUTE THE DIFFERENTIALS
    
    // First we will compute the matrix for sampled data,
    // then for real data and subtract the sampled matrix.
    // Note that weights matrix is stored as column-primary.
    k_gemm(cublasHandle, CblasNoTrans, CblasNoTrans,
            outputsCount, inputSize, 1,
            lr, sOutputs, sInputs, (data_t) 0., weightDiffs);
    
    k_gemm(cublasHandle, CblasNoTrans, CblasNoTrans,
            outputsCount, inputSize, 1,
            lr, outputs, inputs, (data_t) -1., weightDiffs);
    
    // compute differentials for bias
    if (conf.useBias) {
        
        data_t *vdiffs = weightDiffs + genuineWeightsCount;
        data_t *hdiffs = vdiffs + inputSize;
        
        // clear the bias diffs just before working with them
        checkCudaErrors(cudaMemset(vdiffs, 0, (inputSize + outputsCount) * sizeof(data_t)));
        
        // compute bias for visible neurons
        k_axpy(cublasHandle, inputSize, (data_t) lr, inputs, 1, vdiffs, 1);
        k_axpy(cublasHandle, inputSize, (data_t) -lr, sInputs, 1, vdiffs, 1);
        
        // compute bias for hidden neurons
        k_axpy(cublasHandle, outputsCount, (data_t) lr, outputs, 1, hdiffs, 1);
        k_axpy(cublasHandle, outputsCount, (data_t) -lr, sOutputs, 1, hdiffs, 1);
    }
    
    // adjust RBM parameters according to computed diffs
    k_sumVectors(weights, weightDiffs, weightsCount);
}


void RbmLayer::sample_vh_gpu(data_t *inputs, data_t *potentials, data_t *outputs) {
    
    propagateForwardGpu(inputs, potentials, outputs);

    k_generateUniform(*curandGen, randomData, outputsCount);
    k_uniformToCoinFlip(outputs, randomData, outputsCount);
}

void RbmLayer::sample_hv_gpu(data_t *outputs, data_t *potentials, data_t *inputs) {
    
    propagateBackwardGpu(outputs, potentials, inputs);
    
    // Do not sample visible states, use probabilities instead?
    k_generateUniform(*curandGen, randomData, inputSize);
    k_uniformToCoinFlip(inputs, randomData, inputSize);
}

void RbmLayer::gibbs_hvh(int steps) {
    
    // start from 1, since the last step is computed
    // separately using probabilities instead of binary states
    for (int i = 1; i<steps; i++) {
        sample_hv_gpu(sOutputs, svPotentials, sInputs);
        sample_vh_gpu(sInputs, shPotentials, sOutputs);
    }
    
    // The last update should use the probabilities,
    // not the states themselves. This will eliminate
    // the sampling noise during the learning.
    sample_hv_gpu(sOutputs, svPotentials, sInputs);
    propagateForwardGpu(sInputs, shPotentials, sOutputs);
}

void RbmLayer::backwardLastCpu(data_t* expectedOutput) {
    LOG()->error("Backpropagation based on expected output is not implemented in RBM layer. This error happens when RBM layer is the last network layer.");
}

void RbmLayer::backwardLastGpu(data_t* expectedOutput) {
    LOG()->error("Backpropagation based on expected output is not implemented in RBM layer. This error happens when RBM layer is the last network layer.");
}

void RbmLayer::processConfString(string confString) {
    // dummy variable for delimiters
    char sep;
    istringstream iss (confString);
    
    if (!(iss >> conf.outputSize)) {
        LOG()->error("Could not read output size for RBM layer.");
    }
    
    iss >> sep;
    
    if (!(iss >> boolalpha >> conf.isPersistent)) {
        LOG()->warn("Could not read whether to use CD or PCD for RBM layer from configuration. Using CD...");
        conf.isPersistent = false;
    }
    
    iss >> sep;
    
    if (!(iss >> conf.gibbsSteps)) {
        LOG()->error("Could not read number of Gibbs steps for RBM layer, using 1.");
        conf.gibbsSteps = 1;
    }
    
    iss >> sep;
    
    if (!(iss >> lr)) {
        LOG()->warn("Could not read learning rate for RBM layer from configuration. Using default of 0.3.");
        lr = .3;
    }
    
    iss >> sep;
    
    if (!(iss >> boolalpha >> conf.useBias)) {
        LOG()->warn("Could not read bias for RBM layer from configuration. Not using bias...");
        conf.useBias = false;
    }
}

RbmConfig RbmLayer::getConfig() {
    return conf;
}

bool RbmLayer::isPretrainable() {
    return true;
}

static LayerRegister<RbmLayer> reg("Rbm");
