/* 
 * File:   cudaDebugHelpers.h
 * Author: janvojt
 *
 * Created on November 29, 2014, 11:47 PM
 */

#ifndef CUDADEBUGHELPERS_H
#define	CUDADEBUGHELPERS_H

#include "../common.h"
#include "cudaHelpers.h"
#include "../net/Layer.h"

void dumpDeviceInts(const char *flag, const int *dm, const int size);

void dumpDeviceArray(const char *flag, const data_t *dm, const int size);

void compareDeviceArrays(const char *flag, const data_t *dm1, const data_t *dm2, const int size);

void paint2DimageL(const char *flag, Layer *layer);

void paint2Dimage(const char *flag, data_t *data, int size);

void compare2Dimages(const char* flag, data_t *data1, data_t *data2, int size);

void compare(const char *flag, const data_t *dm, const data_t *hm, const int size);

void isNan(const char *flag, const data_t *dm, const int size);

#endif	/* CUDADEBUGHELPERS_H */

