/* 
 * File:   CpuBackpropagationLearner.cpp
 * Author: janvojt
 * 
 * Created on November 20, 2014, 9:40 PM
 */

#include "CpuBackpropagationLearner.h"

#include <iostream>

CpuBackpropagationLearner::CpuBackpropagationLearner(CpuNetwork *network) : BackpropagationLearner(network) {
    allocateCache();
}

CpuBackpropagationLearner::CpuBackpropagationLearner(const CpuBackpropagationLearner& orig) : BackpropagationLearner(orig) {
}

CpuBackpropagationLearner::~CpuBackpropagationLearner() {
}

void CpuBackpropagationLearner::allocateCache() {
    weightDiffs = new double[network->getWeightsOffset(network->getConfiguration()->getLayers())];
    localGradients = new double[network->getAllNeurons()];
    biasDiff = useBias ? new double[network->getAllNeurons()] : NULL;
}

void CpuBackpropagationLearner::computeOutputGradients(double *expectedOutput) {
    int on = network->getOutputNeurons();
    double *localGradient = localGradients + network->getInputOffset(noLayers-1);
    double *output = network->getOutput();
    void (*daf) (double*,double*,int) = network->getConfiguration()->dActivationFnc;
    
    // compute local gradients
    double *dv = new double[network->getOutputNeurons()];
    daf(network->getInputs() + network->getInputOffset(noLayers-1), dv, on);
    for (int i = 0; i<on; i++) {
        localGradient[i] = (output[i] - expectedOutput[i]) * dv[i];
//        LOG()->debug("Local gradient for neuron [%d, %d] : %f.", noLayers, i, localGradient[i]);
    }
    delete[] dv;
}

void CpuBackpropagationLearner::computeWeightDifferentials() {
    
    void (*daf) (double*,double*,int) = network->getConfiguration()->dActivationFnc;
    
    for (int l = noLayers-1; l>0; l--) {
        
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
        for (int i = 0; i<thisNeurons; i++) {
            for (int j = 0; j<nextNeurons; j++) {
                wdiff[i*nextNeurons+j] = -learningRate * nextLocalGradient[j] * thisInput[i];
            }
        }
        
        // COMPUTE BIAS DERIVATIVES for layer l+1
        if (useBias) {
            for (int i = 0; i<nextNeurons; i++) {
                biasDiff[nextInputIdx + i] = -learningRate * nextLocalGradient[i];
            }
        }
        
        // COMPUTE LOCAL GRADIENTS for layer l
        
        // compute derivatives of neuron inputs in layer l
        double *thisInputDerivatives = new double[thisNeurons];
        daf(thisInput, thisInputDerivatives, thisNeurons);
        
        // compute local gradients for layer l
        for (int i = 0; i<thisNeurons; i++) {
            double sumNextGradient = 0;
            for (int j = 0; j<nextNeurons; j++) {
                sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
            }
            thisLocalGradient[i] = sumNextGradient * thisInputDerivatives[i];
//            LOG()->debug("Local gradient for neuron [%d, %d] : %f.", l, i, thisLocalGradient[i]);
        }
        
        delete[] thisInputDerivatives;
    }
}

void CpuBackpropagationLearner::adjustWeights() {
    int wc = network->getWeightsOffset(noLayers);
    double *weights = network->getWeights();
    
    // we should skip the garbage in zero-layer weights
    for(int i = network->getWeightsOffset(1); i<wc; i++) {
        weights[i] += weightDiffs[i];
    }
}

void CpuBackpropagationLearner::adjustBias() {
    double *bias = network->getBiasValues();
    int noNeurons = network->getAllNeurons();
    for (int i = 0; i<noNeurons; i++) {
        bias[i] += biasDiff[i];
    }
}