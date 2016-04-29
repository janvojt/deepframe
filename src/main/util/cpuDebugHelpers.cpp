/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include <cstdlib>
#include <iostream>
#include "cpuDebugHelpers.h"

using namespace std;

void dumpHostArray(const char *flag, float *array, int size) {
    std::cout << flag << std::endl;
    for (int i = 0; i<size; i++) {
        cout << "Dumping host " << flag << ": " << array[i] << endl;
    }
    cout << "-----------------------------" << endl;
}

void printImage(int x, int y, data_t *arr) {
    for (int i = 0; i<x; i++) {
        char sep = ' ';
        cout << sep;
        for (int j = 0; j<y; j++) {
            data_t d = arr[i*x+j];
            if (d<.5) {
                cout << " ";
            } else if (d<.9) {
                cout << "0";
            } else {
                cout << "#";
            }
        }
        cout << endl;
    }
}

void printImageLabels(LabeledDataset *lds) {
    int i = 0;
    lds->reset();
    while (lds->hasNext()) {
        data_t* x = lds->next();
        printImage(28, 28, x);
        cout << endl;
        char sep = ' ';
        int dim = lds->getOutputDimension();
        cout << ++i << ":";
        for (int j = 0; j<dim; j++) {
            cout << sep << x[lds->getInputDimension() + j];
            sep = ',';
        }
        cout << "." << endl;
        cout << endl;
        cout << endl;
    }
}