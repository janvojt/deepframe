/* 
 * File:   ErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on July 20, 2014, 11:46 AM
 */

#include "ErrorComputer.h"

#include "../common.h"

template <typename dType>
ErrorComputer<dType>::ErrorComputer() {
}

template <typename dType>
ErrorComputer<dType>::ErrorComputer(const ErrorComputer& orig) {
}

template <typename dType>
ErrorComputer<dType>::~ErrorComputer() {
}

INSTANTIATE_DATA_CLASS(ErrorComputer);