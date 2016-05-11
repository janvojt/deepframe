/* 
 * File:   cudaDebugHelpers.h
 * Author: janvojt
 *
 * Created on November 29, 2014, 11:47 PM
 * 
 * This file contains helper functions for debugging the CUDA code and memory.
 */

#ifndef CUDADEBUGHELPERS_H
#define	CUDADEBUGHELPERS_H

#include "../common.h"
#include "cudaHelpers.h"
#include "../net/Layer.h"

/**
 * Dumps an array of integers stored in the device memory into stdout.
 * 
 * @param flag use for flagging the dumped array, will be printed in stdout
 * @param dm
 * @param size
 */
void dumpDeviceInts(const char *flag, const int *dm, const int size);

/**
 * Dumps an array of floats stored in the device memory into stdout.
 * 
 * @param flag use for flagging the dumped array, will be printed in stdout
 * @param dm
 * @param size
 */
void dumpDeviceArray(const char *flag, const data_t *dm, const int size);

/**
 * Compares the values of 2 arrays stored in the device memory and
 * prints their respective value into stdout.
 * 
 * @param flag use for flagging the dump, will be printed in stdout
 * @param dm1 pointer to the first array
 * @param dm2 pointer to the second array
 * @param size portion of both arrays to be compared and dumped
 */
void compareDeviceArrays(const char *flag, const data_t *dm1, const data_t *dm2, const int size);

/**
 * Paints the neural activations in a layer as a 2D squared image.
 * 
 * @param flag use for flagging the dump, will be printed in stdout
 * @param layer layer with the neural activations to be painted
 */
void paint2DimageL(const char *flag, Layer *layer);

/**
 * Paints the array values as a 2D squared image.
 * 
 * @param flag use for flagging the dump, will be printed in stdout
 * @param data pointer to the array
 * @param size size of the array
 */
void paint2Dimage(const char *flag, data_t *data, int size);

/**
 * Paints the values from two different arrays as two images
 * next to each other for easy comparison.
 * 
 * @param flag use for flagging the dump, will be printed in stdout
 * @param data1 pointer to the array with data for first image
 * @param data2 pointer to the array with data for second image
 * @param size number of pixels in each image
 */
void compare2Dimages(const char* flag, data_t *data1, data_t *data2, int size);

/**
 * Prints the values of two arrays next to each other for easy comparison.
 * 
 * @param flag use for flagging the dump, will be printed in stdout
 * @param dm pointer to array stored on device memory
 * @param hm pointer to array stored on host memory
 * @param size number of array elements to compare
 */
void compare(const char *flag, const data_t *dm, const data_t *hm, const int size);

/**
 * Looks for a NaN value in the given array.
 * 
 * @param flag use for flagging the dump, will be printed in stdout
 * @param dm pointer to array stored in device memory
 * @param size size of the array
 */
void isNan(const char *flag, const data_t *dm, const int size);

#endif	/* CUDADEBUGHELPERS_H */

