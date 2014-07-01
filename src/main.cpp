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
#include "activationFunctions.h"

using namespace std;

void printOutput(Network *net) {
    cout << "[ ";
    float *out = net->getOutput();
    int oNeurons = net->getOutputNeurons();
    
    cout << *out;
    out++;
    for (int i = 1; i<oNeurons; i++) {
        cout << ", " << *out;
        out++;
    }
    cout << " ]" << endl;
}

void runXorTest(Network *net) {
    
    float input[] = {0, 0};
    net->setInput(input);
    net->run();
    printOutput(net);
    
    float input2[] = {0, 1};
    net->setInput(input2);
    net->run();
    printOutput(net);
    
    float input3[] = {1, 0};
    net->setInput(input3);
    net->run();
    printOutput(net);
    
    float input4[] = {1, 1};
    net->setInput(input4);
    net->run();
    printOutput(net);
    
}

// entry point of the application
int main(int argc, char **argv) {
    NetworkConfiguration *conf = new NetworkConfiguration();

    conf->setLayers(3);
    conf->setNeurons(0, 2);
    conf->setNeurons(1, 3);
    conf->setNeurons(2, 2);
    conf->activationFnc = sigmoidFunction;
    
    Network *net = new Network(conf);
    runXorTest(net);

    delete conf;
    delete net;
    return 0;
}

