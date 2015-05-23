/* 
 * File:   CpuBackpropagationLearner.cpp
 * Author: janvojt
 * 
 * Created on November 20, 2014, 9:40 PM
 */

#include "CpuBackpropagationLearner.h"

#include "../util/cpuDebugHelpers.h"
#include "../common.h"

#include <iostream>

template <typename dType>
CpuBackpropagationLearner<dType>::CpuBackpropagationLearner(CpuNetwork<dType> *network) : BackpropagationLearner<dType>(network) {
    allocateCache();
}

template <typename dType>
CpuBackpropagationLearner<dType>::CpuBackpropagationLearner(const CpuBackpropagationLearner& orig) : BackpropagationLearner<dType>(orig) {
}

template <typename dType>
CpuBackpropagationLearner<dType>::~CpuBackpropagationLearner() {
    delete[] this->weightDiffs;
    delete[] this->localGradients;
    if (this->useBias) delete[] this->biasDiff;
}

template <typename dType>
void CpuBackpropagationLearner<dType>::allocateCache() {
//    this->weightDiffs = new dType[this->network->getWeightsOffset(this->network->getConfiguration()->getLayers())];
//    this->localGradients = new dType[this->network->getAllNeurons()];
//    this->biasDiff = this->useBias ? new dType[this->network->getAllNeurons()] : NULL;
}

template <typename dType>
void CpuBackpropagationLearner<dType>::computeOutputGradients(dType *expectedOutput) {
//    int oNeurons = this->network->getOutputNeurons();
//    dType *localGradient = this->localGradients + this->network->getInputOffset(this->noLayers-1);
//    dType *output = this->network->getInputs() + this->network->getInputOffset(this->noLayers-1);
//    void (*daf) (dType*,dType*,int) = this->network->getConfiguration()->dActivationFnc;
//    
//    // compute local gradients
//    dType *dv = new dType[oNeurons];
//    daf(output, dv, oNeurons);
//    for (int i = 0; i<oNeurons; i++) {
//        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
////        LOG()->debug("Local gradient for neuron [%d, %d] : %f.", noLayers, i, localGradient[i]);
//    }
//    dumpHostArray('o', localGradients, network->getInputOffset(noLayers));
//    delete[] dv;
}

template <typename dType>
void CpuBackpropagationLearner<dType>::computeWeightDifferentials() {
    
    void (*daf) (dType*,dType*,int) = this->network->getConfiguration()->dActivationFnc;
    
    for (int l = this->noLayers-1; l>0; l--) {
        
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
//        for (int i = 0; i<thisNeurons; i++) {
//            for (int j = 0; j<nextNeurons; j++) {
//                wdiff[i*nextNeurons+j] = -this->learningRate * nextLocalGradient[j] * thisInput[i];
//            }
//        }
////        dumpHostArray('w', wdiff, thisNeurons * nextNeurons);
//        
//        // COMPUTE BIAS DERIVATIVES for layer l+1
//        if (this->useBias) {
//            for (int i = 0; i<nextNeurons; i++) {
//                this->biasDiff[nextInputIdx + i] = -this->learningRate * nextLocalGradient[i];
//            }
////            dumpHostArray('c', &biasDiff[nextInputIdx], nextNeurons);
//        }
//        
//        // COMPUTE LOCAL GRADIENTS for layer l
//        
//        // compute derivatives of neuron inputs in layer l
//        dType *thisInputDerivatives = new dType[thisNeurons];
//        daf(thisInput, thisInputDerivatives, thisNeurons);
//        
//        // compute local gradients for layer l
//        for (int i = 0; i<thisNeurons; i++) {
//            dType sumNextGradient = 0;
//            for (int j = 0; j<nextNeurons; j++) {
//                sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
//            }
//            thisLocalGradient[i] = sumNextGradient * thisInputDerivatives[i];
////            LOG()->debug("Local gradient for neuron [%d, %d] : %f.", l, i, thisLocalGradient[i]);
//        }
////        dumpHostArray('l', thisLocalGradient, thisNeurons + nextNeurons);
//        
//        delete[] thisInputDerivatives;
    }
}

template <typename dType>
void CpuBackpropagationLearner<dType>::adjustWeights() {
//    int wc = this->network->getWeightsOffset(this->noLayers);
//    dType *weights = this->network->getWeights();
//    
//    // we should skip the garbage in zero-layer weights
//    for(int i = this->network->getWeightsOffset(1); i<wc; i++) {
//        weights[i] += this->weightDiffs[i];
//    }
//    dumpHostArray('w', weights, network->getWeightsOffset(noLayers));
}

template <typename dType>
void CpuBackpropagationLearner<dType>::adjustBias() {
//    dType *bias = this->network->getBiasValues();
//    int noNeurons = this->network->getAllNeurons();
//    for (int i = 0; i<noNeurons; i++) {
//        bias[i] += this->biasDiff[i];
//    }
//    dumpHostArray('b', bias, network->getWeightsOffset(noLayers));
}

INSTANTIATE_DATA_CLASS(CpuBackpropagationLearner);