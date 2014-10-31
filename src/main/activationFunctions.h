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
void sigmoidFunction(double *x, double *y, int layerSize);

// Derivative of the sigmoid function.
void dSigmoidFunction(double *x, double *y, int layerSize);

// Computes identity function. Essentially, no computation is done.
void identityFunction(double *x, double *y, int layerSize);

// Derivative if the identity function. Always returns 1.
void dIdentityFunction(double *x, double *y, int layerSize);

// Computes hyperbolic tangent for each value in the input array,
// putting the result in the target array. For in-place computation
// it is possible to provide the same pointer for input and target
// to save some memory.
void hyperbolicTangentFunction(double *x, double *y, int layerSize);

// Derivative of the hyperbolic tangent function.
void dHyperbolicTangentFunction(double *x, double *y, int layerSize);

// Looks up activation function in a precomputed table for the input array,
// putting the result in the target array. For in-place computation
// it is possible to provide the same pointer for input and target
// to save some memory.
// See FunctionCache for details of implementation.
void cachedFunction(double *x, double *y, int layerSize);

#endif	/* ACTIVATIONFUNCTIONS_H */

