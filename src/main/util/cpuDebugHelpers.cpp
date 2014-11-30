/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include <cstdlib>
#include <iostream>

using namespace std;

void dumpHostArray(char flag, double *array, int size) {
    for (int i = 0; i<size; i++) {
        cout << "Dumping host " << flag << ": " << array[i] << endl;
    }
    cout << "-----------------------------" << endl;
}