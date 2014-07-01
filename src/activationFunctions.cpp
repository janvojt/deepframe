/* 
 * File:   activationFunctions.cpp
 * Author: janvojt
 * 
 * Created on July 1, 2014, 10:20 PM
 */

#include "activationFunctions.h"

#include <math.h>

// Computes sigmoid function on given neuron potentials.
void sigmoidFunction(float *inputPtr, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *inputPtr = 1 / (1 + exp(-*inputPtr));
        inputPtr++;
    }
}