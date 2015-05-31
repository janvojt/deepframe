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
    cudaFree(this->weightDiffs);
    cudaFree(this->localGradients);
    if (this->useBias) cudaFree(this->biasDiff);
}

void GpuBackpropagationLearner::allocateCache() {
    
//    int dSize = sizeof(data_t);
//    int noWeights = this->network->getWeightsOffset(this->noLayers);
//    int noNeurons = this->network->getAllNeurons();
//    
//    checkCudaErrors(cudaMalloc(&this->weightDiffs, noWeights * dSize));
//    checkCudaErrors(cudaMalloc(&this->localGradients, noNeurons * dSize));
//    if (this->useBias) checkCudaErrors(cudaMalloc(&this->biasDiff, noNeurons * dSize));
}

void GpuBackpropagationLearner::computeOutputGradients(data_t *expectedOutput) {
    
    LOG()->debug("Computing local gradients for output layer.");

//    data_t *localGradient = this->localGradients + this->network->getInputOffset(this->noLayers-1);
//    data_t *output = this->network->getInputs() + this->network->getInputOffset(this->noLayers-1);
//    
//    int oNeurons = this->network->getOutputNeurons();
//    int memSize = oNeurons * sizeof(data_t);
//    data_t *dExpOutput;
//    checkCudaErrors(cudaMalloc(&dExpOutput, memSize));
//    checkCudaErrors(cudaMemcpy(dExpOutput, expectedOutput, memSize, cudaMemcpyHostToDevice));
//    k_computeOutputLocalGradient(output, dExpOutput, localGradient, oNeurons);
//    cudaFree(dExpOutput);
    
//    dumpDeviceArray('o', localGradients, network->getInputOffset(noLayers));
}

void GpuBackpropagationLearner::computeWeightDifferentials() {
    
    for (int l = this->noLayers-1; l>0; l--) {
        
        LOG()->debug("Computing weight differentials between layers %d and %d.", l, l+1);
        
        // INITIALIZE HELPER VARIABLES
//        int thisInputIdx = this->network->getInputOffset(l-1);
//        data_t *thisLocalGradient = this->localGradients + thisInputIdx;
//        int nextInputIdx = this->network->getInputOffset(l);
//        data_t *nextLocalGradient = this->localGradients + nextInputIdx;
//        int thisNeurons = this->network->getConfiguration()->getNeurons(l-1);
//        int nextNeurons = this->network->getConfiguration()->getNeurons(l);
//        data_t *thisInput = this->network->getInputs() + thisInputIdx;
//        data_t *weights = this->network->getWeights() + this->network->getWeightsOffset(l);
//        
//        
//        // COMPUTE TOTAL DERIVATIVES for weights between layer l and l+1
//        data_t *wdiff = this->weightDiffs + this->network->getWeightsOffset(l);
//        k_computeTotalDerivative(thisNeurons, nextNeurons,
//                this->learningRate, thisInput, nextLocalGradient,
//                wdiff);
//    
////        dumpDeviceArray('w', wdiff, thisNeurons * nextNeurons);
//        
//        // COMPUTE BIAS DERIVATIVES for layer l+1
//        if (this->useBias) {
//            k_computeBiasDerivative(
//                    this->learningRate, nextLocalGradient,
//                    &this->biasDiff[nextInputIdx],
//                    nextNeurons);
////            dumpDeviceArray('c', &biasDiff[nextInputIdx], nextNeurons);
//        }
//        
//        // COMPUTE LOCAL GRADIENTS for layer l
//        k_computeHiddenLocalGradient(
//                thisNeurons, nextNeurons,
//                thisInput, weights,
//                thisLocalGradient, nextLocalGradient);
////        dumpDeviceArray('l', thisLocalGradient, thisNeurons + nextNeurons);
    }
}

void GpuBackpropagationLearner::adjustWeights() {
    
    LOG()->debug("Adjusting weights.");
    
//    // we should skip the garbage in zero-layer weights
//    int trim = this->network->getWeightsOffset(1);
//    
//    int wc = this->network->getWeightsOffset(this->noLayers) - trim;
//    data_t *weights = this->network->getWeights();
//    
//    k_sumVectors(weights + trim, this->weightDiffs + trim, wc);
    
//    dumpDeviceArray('w', weights, network->getWeightsOffset(noLayers));
}

void GpuBackpropagationLearner::adjustBias() {
    
    LOG()->debug("Adjusting bias.");
    
//    data_t *bias = this->network->getBiasValues();
//    int noNeurons = this->network->getAllNeurons();
//    
//    k_sumVectors(bias, this->biasDiff, noNeurons);
//    
////    dumpDeviceArray('b', network->getInputs(), network->getInputOffset(noLayers));
}
