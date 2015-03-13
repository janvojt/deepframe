/* 
 * File:   activationFunctions.cpp
 * Author: janvojt
 * 
 * Created on July 1, 2014, 10:20 PM
 */

#include "activationFunctions.h"
#include "FunctionCache.h"

#include <math.h>

template <typename dType>
void sigmoidFunction(dType *x, dType *y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1 / (1 + exp(-*x));
        x++;
        y++;
    }
}
template void sigmoidFunction<float>(float*, float*, int);
template void sigmoidFunction<double>(double*, double*, int);

template <typename dType>
void dSigmoidFunction(dType* x, dType* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = *x * (1 - *x);
        x++;
        y++;
    }
}
template void dSigmoidFunction<float>(float*, float*, int);
template void dSigmoidFunction<double>(double*, double*, int);

template <typename dType>
void identityFunction(dType *x, dType *y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = *x;
        x++;
        y++;
    }
}
template void identityFunction<float>(float*, float*, int);
template void identityFunction<double>(double*, double*, int);

template <typename dType>
void dIdentityFunction(dType* x, dType* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1;
        x++;
        y++;
    }
}
template void dIdentityFunction<float>(float*, float*, int);
template void dIdentityFunction<double>(double*, double*, int);

template <typename dType>
void hyperbolicTangentFunction(dType* x, dType* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = tanh(*x);
        x++;
        y++;
    }
}
template void hyperbolicTangentFunction<float>(float*, float*, int);
template void hyperbolicTangentFunction<double>(double*, double*, int);

template <typename dType>
void dHyperbolicTangentFunction(dType* x, dType* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        *y = 1 - (*x * *x);
        x++;
        y++;
    }
}
template void dHyperbolicTangentFunction<float>(float*, float*, int);
template void dHyperbolicTangentFunction<double>(double*, double*, int);

template <typename dType>
void cachedFunction(dType* x, dType* y, int layerSize) {
    FunctionCache<dType>::getInstance()->compute(x, y, layerSize);
}
template void cachedFunction<float>(float*, float*, int);
template void cachedFunction<double>(double*, double*, int);
