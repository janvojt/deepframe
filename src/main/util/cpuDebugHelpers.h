/* 
 * File:   cpuDebugHelpers.h
 * Author: janvojt
 *
 * Created on November 30, 2014, 12:36 AM
 * 
 * This file declares helper function purposed for debugging the CPU code.
 */

#ifndef CPUDEBUGHELPERS_H
#define	CPUDEBUGHELPERS_H

#include "../common.h"
#include "../ds/LabeledDataset.h"

/**
 * Dump array values stored in host memory.
 * 
 * @param flag use for flagging the dumped array, will be printed in stdout
 * @param array pointer to the array to be dumped
 * @param size size of the array
 */
void dumpHostArray(const char *flag, data_t *array, int size);

/**
 * Paints the image using characters into stdout. The values
 * greater or equal to 1 are painted black (#), less than a half
 * are painted white (space).
 * 
 * @param x width of the image
 * @param y height of the image
 * @param arr image data
 */
void printImage(int x, int y, data_t *arr);

/**
 * Paints the images in a dataset and prints out its label to stdout.
 * 
 * @param lds image dataset
 */
void printImageLabels(LabeledDataset *lds);

#endif	/* CPUDEBUGHELPERS_H */

