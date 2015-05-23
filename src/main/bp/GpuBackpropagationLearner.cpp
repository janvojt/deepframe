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


template <typename dType>
GpuBackpropagationLearner<dType>::GpuBackpropagationLearner(GpuNetwork<dType> * network) : BackpropagationLearner<dType>(network) {
    allocateCache();
}

template <typename dType>
GpuBackpropagationLearner<dType>::GpuBackpropagationLearner(const GpuBackpropagationLearner& orig) : BackpropagationLearner<dType>(orig) {
}

template <typename dType>
GpuBackpropagationLearner<dType>::~GpuBackpropagationLearner() {
    cudaFree(this->weightDiffs);
    cudaFree(this->localGradients);
    if (this->useBias) cudaFree(this->biasDiff);
}

template <typename dType>
void GpuBackpropagationLearner<dType>::allocateCache() {
    
//    int dSize = sizeof(dType);
//    int noWeights = this->network->getWeightsOffset(this->noLayers);
//    int noNeurons = this->network->getAllNeurons();
//    
//    checkCudaErrors(cudaMalloc(&this->weightDiffs, noWeights * dSize));
//    checkCudaErrors(cudaMalloc(&this->localGradients, noNeurons * dSize));
//    if (this->useBias) checkCudaErrors(cudaMalloc(&this->biasDiff, noNeurons * dSize));
}

template <typename dType>
void GpuBackpropagationLearner<dType>::computeOutputGradients(dType *expectedOutput) {
    
    LOG()->debug("Computing local gradients for output layer.");

//    dType *localGradient = this->localGradients + this->network->getInputOffset(this->noLayers-1);
//    dType *output = this->network->getInputs() + this->network->getInputOffset(this->noLayers-1);
//    
//    int oNeurons = this->network->getOutputNeurons();
//    int memSize = oNeurons * sizeof(dType);
//    dType *dExpOutput;
//    checkCudaErrors(cudaMalloc(&dExpOutput, memSize));
//    checkCudaErrors(cudaMemcpy(dExpOutput, expectedOutput, memSize, cudaMemcpyHostToDevice));
//    k_computeOutputLocalGradient(output, dExpOutput, localGradient, oNeurons);
//    cudaFree(dExpOutput);
    
//    dumpDeviceArray('o', localGradients, network->getInputOffset(noLayers));
}

template <typename dType>
void GpuBackpropagationLearner<dType>::computeWeightDifferentials() {
    
    for (int l = this->noLayers-1; l>0; l--) {
        
        LOG()->debug("Computing weight differentials between layers %d and %d.", l, l+1);
        
        // INITIALIZE HELPER VARIABLES
//        int thisInputIdx = this->network->getInputOffset(l-1);
//        dType *thisLocalGradient = this->localGradients + thisInputIdx;
//        int nextInputIdx = this->network->getInputOffset(l);
//        dType *nextLocalGradient = this->localGradients + nextInputIdx;
//        int thisNeurons = this->network->getConfiguration()->getNeurons(l-1);
//        int nextNeurons = this->network->getConfiguration()->getNeurons(l);
//        dType *thisInput = this->network->getInputs() + thisInputIdx;
//        dType *weights = this->network->getWeights() + this->network->getWeightsOffset(l);
//        
//        
//        // COMPUTE TOTAL DERIVATIVES for weights between layer l and l+1
//        dType *wdiff = this->weightDiffs + this->network->getWeightsOffset(l);
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

template <typename dType>
void GpuBackpropagationLearner<dType>::adjustWeights() {
    
    LOG()->debug("Adjusting weights.");
    
//    // we should skip the garbage in zero-layer weights
//    int trim = this->network->getWeightsOffset(1);
//    
//    int wc = this->network->getWeightsOffset(this->noLayers) - trim;
//    dType *weights = this->network->getWeights();
//    
//    k_sumVectors(weights + trim, this->weightDiffs + trim, wc);
    
//    dumpDeviceArray('w', weights, network->getWeightsOffset(noLayers));
}

template <typename dType>
void GpuBackpropagationLearner<dType>::adjustBias() {
    
    LOG()->debug("Adjusting bias.");
    
//    dType *bias = this->network->getBiasValues();
//    int noNeurons = this->network->getAllNeurons();
//    
//    k_sumVectors(bias, this->biasDiff, noNeurons);
//    
////    dumpDeviceArray('b', network->getInputs(), network->getInputOffset(noLayers));
}

INSTANTIATE_DATA_CLASS(GpuBackpropagationLearner);