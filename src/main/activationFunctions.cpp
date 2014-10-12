/* 
 * File:   activationFunctions.cpp
 * Author: janvojt
 * 
 * Created on July 1, 2014, 10:20 PM
 */

#include "activationFunctions.h"

#include <math.h>

void sigmoidFunction(double *inputPtr, double *targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = 1 / (1 + exp(-*inputPtr));
        inputPtr++;
        targetPtr++;
    }
}

void dSigmoidFunction(double* inputPtr, double* targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = *inputPtr * (1 - *inputPtr);
        inputPtr++;
        targetPtr++;
    }
}

void identityFunction(double *inputPtr, double *targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = *inputPtr;
        inputPtr++;
        targetPtr++;
    }
}

void dIdentityFunction(double* inputPtr, double* targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = 1;
        inputPtr++;
        targetPtr++;
    }
}

void hyperbolicTangentFunction(double* inputPtr, double* targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = tanh(*inputPtr);
        inputPtr++;
        targetPtr++;
    }
}

void dHyperbolicTangentFunction(double* inputPtr, double* targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = 1 - (*inputPtr * *inputPtr);
        inputPtr++;
        targetPtr++;
    }
}
