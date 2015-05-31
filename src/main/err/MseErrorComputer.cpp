/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on July 20, 2014, 11:48 AM
 */

#include "MseErrorComputer.h"

#include "../common.h"

#include <math.h>

MseErrorComputer::MseErrorComputer() {
}

MseErrorComputer::MseErrorComputer(const MseErrorComputer& orig) {
}

MseErrorComputer::~MseErrorComputer() {
}

data_t MseErrorComputer::compute(Network* net, data_t* expectedOutput) {
    int oNeurons = net->getOutputNeurons();
    data_t *output = net->getOutput();
    data_t mse = 0;
    for (int i = 0; i<oNeurons; i++) {
        mse += pow(output[i] - expectedOutput[i], 2);
    }
    mse = mse / oNeurons;
    return mse;
}
