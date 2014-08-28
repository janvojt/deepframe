/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on July 20, 2014, 11:48 AM
 */

#include "MseErrorComputer.h"
#include "Network.h"
#include <math.h>

MseErrorComputer::MseErrorComputer() {
}

MseErrorComputer::MseErrorComputer(const MseErrorComputer& orig) {
}

MseErrorComputer::~MseErrorComputer() {
}

double MseErrorComputer::compute(Network* net, double* expectedOutput) {
    int oNeurons = net->getOutputNeurons();
    double *output = net->getOutput();
    double mse = 0;
    for (int i = 0; i<oNeurons; i++) {
        mse += pow(output[i] - expectedOutput[i], 2);
    }
    mse = mse / 2;
    return mse;
}
