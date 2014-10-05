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
void sigmoidFunction(double *inputPtr, double *targetPtr, int layerSize);

// Derivative of the sigmoid function.
void dSigmoidFunction(double *inputPtr, double *targetPtr, int layerSize);

// Gradient for the sigmoid function.
void gSigmoidFunction(double *inputPtr, double *targetPtr, int layerSize);

#endif	/* ACTIVATIONFUNCTIONS_H */
