/* 
 * File:   cpuDebugHelpers.h
 * Author: janvojt
 *
 * Created on November 30, 2014, 12:36 AM
 */

#ifndef CPUDEBUGHELPERS_H
#define	CPUDEBUGHELPERS_H

#include "../common.h"
#include "../ds/LabeledDataset.h"

void dumpHostArray(const char *flag, float *array, int size);

void printImage(int x, int y, data_t *arr);

void printImageLabels(LabeledDataset *lds);

#endif	/* CPUDEBUGHELPERS_H */

