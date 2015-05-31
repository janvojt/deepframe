/* 
 * File:   common.h
 * Author: janvojt
 *
 * Created on March 10, 2015, 10:20 PM
 */

#ifndef COMMON_H
#define	COMMON_H

// Define the data type
// that will be used for decimal values (weights, bias, ...).
// Uncomment to use 64 bit integers.
//#define USE_64BIT_PRECISION

#ifdef USE_64BIT_PRECISION
typedef double data_t;
#else
typedef float data_t;
#endif

// Instantiate a class with float and double specifications.
#define INSTANTIATE_DATA_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

#endif	/* COMMON_H */

