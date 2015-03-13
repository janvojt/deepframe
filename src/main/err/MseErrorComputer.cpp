/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on July 20, 2014, 11:48 AM
 */

#include "MseErrorComputer.h"

#include "../common.h"

#include <math.h>

template <typename dType>
MseErrorComputer<dType>::MseErrorComputer() {
}

template <typename dType>
MseErrorComputer<dType>::MseErrorComputer(const MseErrorComputer& orig) {
}

template <typename dType>
MseErrorComputer<dType>::~MseErrorComputer() {
}

template <typename dType>
dType MseErrorComputer<dType>::compute(Network<dType>* net, dType* expectedOutput) {
    int oNeurons = net->getOutputNeurons();
    dType *output = net->getOutput();
    dType mse = 0;
    for (int i = 0; i<oNeurons; i++) {
        mse += pow(output[i] - expectedOutput[i], 2);
    }
    mse = mse / oNeurons;
    return mse;
}

INSTANTIATE_DATA_CLASS(MseErrorComputer);