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

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

RbmLayer::RbmLayer() {
}

RbmLayer::RbmLayer(const RbmLayer& orig) {
}

RbmLayer::~RbmLayer() {
}

void RbmLayer::setup(string confString) {
    
    processConfString(confString);
    
    outputsCount = conf.outputSize;
    
    inputSize = previousLayer->getOutputsCount();
    genuineWeightsCount = inputSize * outputsCount;
    
    if (conf.useBias) {
        weightsCount = genuineWeightsCount + inputSize + outputsCount;
    } else {
        weightsCount = genuineWeightsCount;
    }
    
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
    
    // store for computing parallel array reduction
    checkCudaErrors(cudaMalloc(&tempReduce, outputsCount * sizeof(data_t)));
}

void RbmLayer::forwardCpu() {
    LOG()->error("Forward run on CPU is not yet implemented for RBM layer.");
}

void RbmLayer::forwardGpu() {
    propagateForwardGpu(previousLayer->getOutputs(), ohPotentials, outputs);
}

void RbmLayer::backwardCpu() {
    LOG()->error("Backward run on CPU is not yet implemented for RBM layer.");
}

void RbmLayer::backwardGpu() {
    propagateBackwardGpu(outputs, ovPotentials, sInputs);
}

void RbmLayer::propagateForwardCpu(data_t* visibles, data_t* potentials, data_t* hiddens) {

}

void RbmLayer::propagateForwardGpu(data_t* visibles, data_t* potentials, data_t* hiddens) {
    
    data_t *hbias = weights + inputSize * outputsCount + inputSize;
    
    //note cuBLAS is column primary!
    //need to transpose the order
    k_gemm(this->cublasHandle,
            CblasTrans, CblasNoTrans,
            /*n*/outputsCount,/*m*/ 1,/*k*/ inputSize,
            (data_t) 1., weights,
            visibles, (data_t) 0., potentials);

    if (conf.useBias) {
        k_sumVectors(potentials, hbias, outputsCount);
    }
    
    // TODO rework sigmoid so it can do out-place computation
    int memSize = outputsCount * sizeof(data_t);
    checkCudaErrors(cudaMemcpy(hiddens, potentials, memSize, cudaMemcpyDeviceToDevice));
    k_computeSigmoid(hiddens, outputsCount);
}

void RbmLayer::propagateBackwardCpu(data_t* hiddens, data_t* potentials, data_t* visibles) {

}

void RbmLayer::propagateBackwardGpu(data_t* hiddens, data_t* potentials, data_t* visibles) {
    
    data_t *vbias = weights + inputSize * outputsCount;
    
    //note cuBLAS is column primary!
    //need to transpose the order
    k_gemm(this->cublasHandle,
            CblasTrans, CblasNoTrans,
            /*n*/inputSize,/*m*/ 1,/*k*/ outputsCount,
            (data_t) 1., weights,
            hiddens, (data_t) 0., potentials);

    if (conf.useBias) {
        k_sumVectors(potentials, vbias, inputSize);
    }
    
    int memSize = inputSize * sizeof(data_t);
    checkCudaErrors(cudaMemcpy(visibles, potentials, memSize, cudaMemcpyDeviceToDevice));
    k_computeSigmoid(visibles, inputSize);
}

void RbmLayer::pretrainCpu() {

}

void RbmLayer::pretrainGpu() {
    
    data_t *inputs =  previousLayer->getOutputs();
    
    // single forward run will compute original results
    // needed to compute the differentials
    forwardGpu();
    
    // Reset the parameters sampled from previous training
    if (!conf.isPersistent || !samplesInitialized) {
        int memSize = inputSize * sizeof(data_t);
        checkCudaErrors(cudaMemcpy(sInputs, inputs, memSize, cudaMemcpyDeviceToDevice));
        sample_vh_gpu();
    }
    
    // perform CD-k
    gibbs_hvh(conf.gibbsSteps);
    
    // COMPUTE THE DIFFERENTIALS
    
    // First we will compute the matrix for sampled data,
    // then for real data and subtract the sampled matrix.
    k_gemm(cublasHandle, CblasNoTrans, CblasNoTrans,
            outputsCount, inputSize, 1,
            lr, sOutputs, sInputs, (data_t) 0., weightDiffs);
    
    k_gemm(cublasHandle, CblasNoTrans, CblasNoTrans,
            outputsCount, inputSize, 1,
            lr, outputs, inputs, (data_t) -1., weightDiffs);
    
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


void RbmLayer::sample_vh_gpu() {
    
    propagateForwardGpu(sInputs, shPotentials, sOutputs);

    k_generateUniform(*curandGen, randomData, outputsCount);
    k_uniformToCoinFlip(sOutputs, randomData, outputsCount);
}

void RbmLayer::sample_hv_gpu() {
    
    propagateBackwardGpu(sOutputs, svPotentials, sInputs);
    
    k_generateUniform(*curandGen, randomData, inputSize);
    k_uniformToCoinFlip(sInputs, randomData, inputSize);
}

void RbmLayer::gibbs_hvh(int steps) {
    
    for (int i = 0; i<steps; i++) {
        sample_hv_gpu();
        sample_vh_gpu();
    }
}

void RbmLayer::gibbs_vhv(int steps) {
    
    for (int i = 0; i<steps; i++) {
        sample_vh_gpu();
        sample_hv_gpu();
    }
}

data_t RbmLayer::freeEnergy() {
    
    data_t *vbias = weights + inputSize * outputsCount;

    data_t vbiasTerm = 0;
    k_dotProduct(cublasHandle, inputSize, sInputs, 1, vbias, 1, &vbiasTerm);
    
    data_t hiddenTerm = k_logPlusExpReduce(1., shPotentials, tempReduce, outputsCount);
    
    return -hiddenTerm - vbiasTerm;
}

void RbmLayer::costUpdates() {
    
    data_t *inputs = previousLayer->getOutputs();
    
    // Reset the parameters sampled from previous training
    if (!conf.isPersistent || !samplesInitialized) {
        int memSize = inputSize * sizeof(data_t);
        checkCudaErrors(cudaMemcpy(sInputs, inputs, memSize, cudaMemcpyDeviceToDevice));
        sample_vh_gpu();
    } else {
        // TODO ???
    }
    
    data_t oEnergy = freeEnergy();
    
    gibbs_hvh(conf.gibbsSteps);
    
    data_t cost = oEnergy - freeEnergy();
    // TODO what to do with cost?
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
