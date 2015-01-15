/* 
 * File:   cudaDebugHelpers.h
 * Author: janvojt
 *
 * Created on November 29, 2014, 11:47 PM
 */

#ifndef CUDADEBUGHELPERS_H
#define	CUDADEBUGHELPERS_H

#include "cudaHelpers.h"

void dumpDeviceArray(char flag, double *dm, int size);

void compare(char flag, double *dm, double *hm, int size);

#endif	/* CUDADEBUGHELPERS_H */

