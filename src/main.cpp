/* 
 * File:   main.cpp
 * Author: janvojt
 *
 * Created on May 29, 2014, 9:53 PM
 */

#include <cstdlib>
#include <iostream>

#include "NetworkConfiguration.h"
#include "Network.h"

using namespace std;

// entry point of the application
int main(int argc, char** argv) {
    NetworkConfiguration *conf = new NetworkConfiguration();

    conf->setLayers(3);
    conf->setNeurons(1, 2);
    conf->setNeurons(2, 3);
    conf->setNeurons(3, 2);
    
    Network* net = new Network(conf);

    delete conf;
    delete net;
    return 0;
}

