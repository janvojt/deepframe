/* 
 * File:   GpuBackpropagationLearner.cpp
 * Author: janvojt
 * 
 * Created on November 20, 2014, 11:23 PM
 */

#include "GpuBackpropagationLearner.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"



// TODO temporary function for debugging purposes
void printarr(char flag, double *dm, int size) {
    double *hdm = new double[size];
    checkCudaErrors(cudaMemcpy(hdm, dm, sizeof(double) * size, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i<size; i++) {
        std::cout << "Dumping device " << flag << ": " << hdm[i] << std::endl;
    }
    std::cout << "-----------------------------" << std::endl;
    
    delete[] hdm;
}

// Compute A = A + B.
__global__
void sumVectors(double *dA, double *dB) {
    dA[threadIdx.x] += dB[threadIdx.x];
}

__global__
void computeOutputLocalGradient(double *actualOutput, double *expectedOutput, double *localGradient) {
    int i = threadIdx.x;
    double derivative = actualOutput[i] * (1.0 - actualOutput[i]);
    localGradient[i] = (actualOutput[i] - expectedOutput[i]) * derivative;
}

__global__
void computeTotalDerivative(double learningRate, int nextNeurons,
        double *thisInput, double *nextLocalGradient,
        double *weightDiffs) {
    
    int i = threadIdx.x;
    int j = threadIdx.y;

    weightDiffs[i*nextNeurons+j] = -learningRate * nextLocalGradient[j] * thisInput[i];
}

__global__
void computeBiasDerivative(double learningRate, double *nextLocalGradient,
        double *biasDiffs) {
    
    int i = threadIdx.x;
    
    biasDiffs[i] = -learningRate * nextLocalGradient[i];
}

__global__
void computeHiddenLocalGradient(int nextNeurons,
        double *thisInput, double *weights,
        double *thisLocalGradient, double *nextLocalGradient) {
    
    int i = threadIdx.x;
    
    double derivative = thisInput[i] * (1.0 - thisInput[i]);
    
    double sumNextGradient = 0;
    for (int j = 0; j<nextNeurons; j++) {
        sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
    }
    thisLocalGradient[i] = sumNextGradient * derivative;
}

GpuBackpropagationLearner::GpuBackpropagationLearner(GpuNetwork * network) : BackpropagationLearner(network) {
    allocateCache();
}

GpuBackpropagationLearner::GpuBackpropagationLearner(const GpuBackpropagationLearner& orig) : BackpropagationLearner(orig) {
}

GpuBackpropagationLearner::~GpuBackpropagationLearner() {
    cudaFree(weightDiffs);
    cudaFree(localGradients);
    if (useBias) cudaFree(biasDiff);
}

void GpuBackpropagationLearner::allocateCache() {
    
    int dSize = sizeof(double);
    int noWeights = network->getWeightsOffset(noLayers);
    int noNeurons = network->getAllNeurons();
    
    checkCudaErrors(cudaMalloc(&weightDiffs, noWeights * dSize));
    checkCudaErrors(cudaMalloc(&localGradients, noNeurons * dSize));
    if (useBias) checkCudaErrors(cudaMalloc(&biasDiff, noNeurons * dSize));
}

void GpuBackpropagationLearner::computeOutputGradients(double *expectedOutput) {
    
    LOG()->debug("Computing local gradients for output layer.");

    double *localGradient = localGradients + network->getInputOffset(noLayers-1);
    double *output = network->getInputs() + network->getInputOffset(noLayers-1);
    
    int oNeurons = network->getOutputNeurons();
    int memSize = oNeurons * sizeof(double);
    double *dExpOutput;
    checkCudaErrors(cudaMalloc(&dExpOutput, memSize));
    checkCudaErrors(cudaMemcpy(dExpOutput, expectedOutput, memSize, cudaMemcpyHostToDevice));
    computeOutputLocalGradient<<<1,oNeurons>>>(output, dExpOutput, localGradient);
    
//    printarr('o', localGradients, network->getInputOffset(noLayers));
}

void GpuBackpropagationLearner::computeWeightDifferentials() {
    
    for (int l = noLayers-1; l>0; l--) {
        
        LOG()->debug("Computing weight differentials between layers %d and %d.", l, l+1);
        
        // INITIALIZE HELPER VARIABLES
        int thisInputIdx = network->getInputOffset(l-1);
        double *thisLocalGradient = localGradients + thisInputIdx;
        int nextInputIdx = network->getInputOffset(l);
        double *nextLocalGradient = localGradients + nextInputIdx;
        int thisNeurons = network->getConfiguration()->getNeurons(l-1);
        int nextNeurons = network->getConfiguration()->getNeurons(l);
        double *thisInput = network->getInputs() + thisInputIdx;
        double *weights = network->getWeights() + network->getWeightsOffset(l);
        
        
        // COMPUTE TOTAL DERIVATIVES for weights between layer l and l+1
        double *wdiff = weightDiffs + network->getWeightsOffset(l);
        int totBlocks = 1;
        dim3 totTpb(thisNeurons, nextNeurons);
        computeTotalDerivative<<<totBlocks,totTpb>>>(
                learningRate, nextNeurons,
                thisInput, nextLocalGradient, wdiff);
    
//        printarr('w', wdiff, thisNeurons * nextNeurons);
        
        // COMPUTE BIAS DERIVATIVES for layer l+1
        if (useBias) {
            int biasBlocks = 1;
            computeBiasDerivative<<<biasBlocks,nextNeurons>>>(
                    learningRate, nextLocalGradient,
                    &biasDiff[nextInputIdx]);
//            printarr('b', &biasDiff[nextInputIdx], nextNeurons);
        }
        
        // COMPUTE LOCAL GRADIENTS for layer l
        int locBlocks = 1;
        computeHiddenLocalGradient<<<locBlocks,thisNeurons>>>(nextNeurons, thisInput, weights,
                thisLocalGradient, nextLocalGradient);
//        printarr('l', thisLocalGradient, thisNeurons * nextNeurons);
    }
}

void GpuBackpropagationLearner::adjustWeights() {
    
    LOG()->debug("Adjusting weights.");
    
    // we should skip the garbage in zero-layer weights
    int trim = network->getWeightsOffset(1);
    
    int wc = network->getWeightsOffset(noLayers) - trim;
    double *weights = network->getWeights();
    
    sumVectors<<<1,wc>>>(weights + trim, weightDiffs + trim);
    
//    printarr('w', weights, network->getWeightsOffset(noLayers));
}

void GpuBackpropagationLearner::adjustBias() {
    
    LOG()->debug("Adjusting bias.");
//    printarr('b', network->getInputs(), network->getInputOffset(noLayers));
    
    double *bias = network->getBiasValues();
    int noNeurons = network->getAllNeurons();
    
    sumVectors<<<1,noNeurons>>>(bias, biasDiff);
}

