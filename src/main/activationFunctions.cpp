/* 
 * File:   activationFunctions.cpp
 * Author: janvojt
 * 
 * Created on July 1, 2014, 10:20 PM
 */

#include "activationFunctions.h"
#include "FunctionCache.h"

#include <math.h>

void sigmoidFunction(data_t *x, data_t *y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1 / (1 + exp(-*x));
        x++;
        y++;
    }
}

void dSigmoidFunction(data_t* x, data_t* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = *x * (1 - *x);
        x++;
        y++;
    }
}

void identityFunction(data_t *x, data_t *y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = *x;
        x++;
        y++;
    }
}

void dIdentityFunction(data_t* x, data_t* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1;
        x++;
        y++;
    }
}

void hyperbolicTangentFunction(data_t* x, data_t* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = tanh(*x);
        x++;
        y++;
    }
}

void dHyperbolicTangentFunction(data_t* x, data_t* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1 - (*x * *x);
        x++;
        y++;
    }
}

void cachedFunction(data_t* x, data_t* y, int layerSize) {
    FunctionCache::getInstance()->compute(x, y, layerSize);
}
