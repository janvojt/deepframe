/* 
 * File:   activationFunctions.cpp
 * Author: janvojt
 * 
 * Created on July 1, 2014, 10:20 PM
 */

#include "activationFunctions.h"
#include "FunctionCache.h"

#include <math.h>

void sigmoidFunction(double *x, double *y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1 / (1 + exp(-*x));
        x++;
        y++;
    }
}

void dSigmoidFunction(double* x, double* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = *x * (1 - *x);
        x++;
        y++;
    }
}

void identityFunction(double *x, double *y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = *x;
        x++;
        y++;
    }
}

void dIdentityFunction(double* x, double* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1;
        x++;
        y++;
    }
}

void hyperbolicTangentFunction(double* x, double* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = tanh(*x);
        x++;
        y++;
    }
}

void dHyperbolicTangentFunction(double* x, double* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1 - (*x * *x);
        x++;
        y++;
    }
}

void cachedFunction(double* x, double* y, int layerSize) {
    FunctionCache::getInstance()->compute(x, y, layerSize);
}
