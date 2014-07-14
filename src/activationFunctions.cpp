/* 
 * File:   activationFunctions.cpp
 * Author: janvojt
 * 
 * Created on July 1, 2014, 10:20 PM
 */

#include "activationFunctions.h"

#include <math.h>

void sigmoidFunction(float *inputPtr, float *targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = 1 / (1 + exp(-*inputPtr));
        inputPtr++;
        targetPtr++;
    }
}

void dSigmoidFunction(float* inputPtr, float* targetPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *targetPtr = exp(*inputPtr) / pow(1 + exp(*inputPtr), 2);
        inputPtr++;
        targetPtr++;
    }
}