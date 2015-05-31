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

CpuBackpropagationLearner::CpuBackpropagationLearner(CpuNetwork *network) : BackpropagationLearner(network) {
    allocateCache();
}

CpuBackpropagationLearner::CpuBackpropagationLearner(const CpuBackpropagationLearner& orig) : BackpropagationLearner(orig) {
}

CpuBackpropagationLearner::~CpuBackpropagationLearner() {
    delete[] this->weightDiffs;
    delete[] this->localGradients;
    if (this->useBias) delete[] this->biasDiff;
}

void CpuBackpropagationLearner::allocateCache() {
//    this->weightDiffs = new data_t[this->network->getWeightsOffset(this->network->getConfiguration()->getLayers())];
//    this->localGradients = new data_t[this->network->getAllNeurons()];
//    this->biasDiff = this->useBias ? new data_t[this->network->getAllNeurons()] : NULL;
}

void CpuBackpropagationLearner::computeOutputGradients(data_t *expectedOutput) {
//    int oNeurons = this->network->getOutputNeurons();
//    data_t *localGradient = this->localGradients + this->network->getInputOffset(this->noLayers-1);
//    data_t *output = this->network->getInputs() + this->network->getInputOffset(this->noLayers-1);
//    void (*daf) (data_t*,data_t*,int) = this->network->getConfiguration()->dActivationFnc;
//    
//    // compute local gradients
//    data_t *dv = new data_t[oNeurons];
//    daf(output, dv, oNeurons);
//    for (int i = 0; i<oNeurons; i++) {
//        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
////        LOG()->debug("Local gradient for neuron [%d, %d] : %f.", noLayers, i, localGradient[i]);
//    }
//    dumpHostArray('o', localGradients, network->getInputOffset(noLayers));
//    delete[] dv;
}

void CpuBackpropagationLearner::computeWeightDifferentials() {
    
    void (*daf) (data_t*,data_t*,int) = this->network->getConfiguration()->dActivationFnc;
    
    for (int l = this->noLayers-1; l>0; l--) {
        
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
//        data_t *thisInputDerivatives = new data_t[thisNeurons];
//        daf(thisInput, thisInputDerivatives, thisNeurons);
//        
//        // compute local gradients for layer l
//        for (int i = 0; i<thisNeurons; i++) {
//            data_t sumNextGradient = 0;
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

void CpuBackpropagationLearner::adjustWeights() {
//    int wc = this->network->getWeightsOffset(this->noLayers);
//    data_t *weights = this->network->getWeights();
//    
//    // we should skip the garbage in zero-layer weights
//    for(int i = this->network->getWeightsOffset(1); i<wc; i++) {
//        weights[i] += this->weightDiffs[i];
//    }
//    dumpHostArray('w', weights, network->getWeightsOffset(noLayers));
}

void CpuBackpropagationLearner::adjustBias() {
//    data_t *bias = this->network->getBiasValues();
//    int noNeurons = this->network->getAllNeurons();
//    for (int i = 0; i<noNeurons; i++) {
//        bias[i] += this->biasDiff[i];
//    }
//    dumpHostArray('b', bias, network->getWeightsOffset(noLayers));
}
