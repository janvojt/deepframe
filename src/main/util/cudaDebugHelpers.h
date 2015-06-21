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

void dumpDeviceArray(char flag, data_t *dm, int size);

void compare(char flag, data_t *dm, data_t *hm, int size);

#endif	/* CUDADEBUGHELPERS_H */

