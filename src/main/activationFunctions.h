/* 
 * File:   activationFunctions.h
 * Author: janvojt
 *
 * Created on July 1, 2014, 10:36 PM
 */

#ifndef ACTIVATIONFUNCTIONS_H
#define	ACTIVATIONFUNCTIONS_H

// Computes sigmoid function for each value in the input array,
// putting the result in the target array. For in-place computation
// it is possible to provide the same pointer for input and target
// to save some memory.
template <typename dType>
void sigmoidFunction(dType *x, dType *y, int layerSize);

// Derivative of the sigmoid function.
template <typename dType>
void dSigmoidFunction(dType *x, dType *y, int layerSize);

// Computes identity function. Essentially, no computation is done.
template <typename dType>
void identityFunction(dType *x, dType *y, int layerSize);

// Derivative if the identity function. Always returns 1.
template <typename dType>
void dIdentityFunction(dType *x, dType *y, int layerSize);

// Computes hyperbolic tangent for each value in the input array,
// putting the result in the target array. For in-place computation
// it is possible to provide the same pointer for input and target
// to save some memory.
template <typename dType>
void hyperbolicTangentFunction(dType *x, dType *y, int layerSize);

// Derivative of the hyperbolic tangent function.
template <typename dType>
void dHyperbolicTangentFunction(dType *x, dType *y, int layerSize);

// Looks up activation function in a precomputed table for the input array,
// putting the result in the target array. For in-place computation
// it is possible to provide the same pointer for input and target
// to save some memory.
// See FunctionCache for details of implementation.
template <typename dType>
void cachedFunction(dType *x, dType *y, int layerSize);

#endif	/* ACTIVATIONFUNCTIONS_H */

