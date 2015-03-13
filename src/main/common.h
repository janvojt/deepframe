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
#define DATA_TYPE float

// Instantiate a class with float and double specifications.
#define INSTANTIATE_DATA_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

#endif	/* COMMON_H */

