/* 
 * File:   main.cpp
 * Author: janvojt
 *
 * Created on May 29, 2014, 9:53 PM
 */

#include <cstdlib>
#include <iostream>

#include "NetworkConfiguration.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    NetworkConfiguration* conf = new NetworkConfiguration();

    conf->setLayers(3);
    conf->setNeurons(0, 2);

    delete conf;
    return 0;
}

