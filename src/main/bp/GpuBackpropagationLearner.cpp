/* 
 * File:   GpuBackpropagationLearner.cpp
 * Author: janvojt
 * 
 * Created on November 20, 2014, 11:23 PM
 */

#include "GpuBackpropagationLearner.h"
#include "../util/cudaHelpers.h"
#include "../util/cudaDebugHelpers.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"


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
    k_computeOutputLocalGradient(output, dExpOutput, localGradient, oNeurons);
    cudaFree(dExpOutput);
    
//    dumpDeviceArray('o', localGradients, network->getInputOffset(noLayers));
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
        k_computeTotalDerivative(thisNeurons, nextNeurons,
                learningRate, thisInput, nextLocalGradient,
                wdiff);
    
//        dumpDeviceArray('w', wdiff, thisNeurons * nextNeurons);
        
        // COMPUTE BIAS DERIVATIVES for layer l+1
        if (useBias) {
            k_computeBiasDerivative(
                    learningRate, nextLocalGradient,
                    &biasDiff[nextInputIdx],
                    nextNeurons);
//            dumpDeviceArray('c', &biasDiff[nextInputIdx], nextNeurons);
        }
        
        // COMPUTE LOCAL GRADIENTS for layer l
        k_computeHiddenLocalGradient(
                thisNeurons, nextNeurons,
                thisInput, weights,
                thisLocalGradient, nextLocalGradient);
//        dumpDeviceArray('l', thisLocalGradient, thisNeurons + nextNeurons);
    }
}

void GpuBackpropagationLearner::adjustWeights() {
    
    LOG()->debug("Adjusting weights.");
    
    // we should skip the garbage in zero-layer weights
    int trim = network->getWeightsOffset(1);
    
    int wc = network->getWeightsOffset(noLayers) - trim;
    double *weights = network->getWeights();
    
    k_sumVectors(weights + trim, weightDiffs + trim, wc);
    
//    dumpDeviceArray('w', weights, network->getWeightsOffset(noLayers));
}

void GpuBackpropagationLearner::adjustBias() {
    
    LOG()->debug("Adjusting bias.");
    
    double *bias = network->getBiasValues();
    int noNeurons = network->getAllNeurons();
    
    k_sumVectors(bias, biasDiff, noNeurons);
    
//    dumpDeviceArray('b', network->getInputs(), network->getInputOffset(noLayers));
}

