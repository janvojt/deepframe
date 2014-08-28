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
#include "BackpropagationLearner.h"
#include "SimpleLabeledDataset.h"
#include "MseErrorComputer.h"

using namespace std;

void printOutput(Network *net) {
    
    // print input
    cout << "[ ";
    double *in = net->getInput();
    int iNeurons = net->getInputNeurons();
    cout << *in++;
    for (int i = 1; i<iNeurons; i++) {
        cout << ", " << *in;
        in++;
    }
    cout << " ] -> ";
    
    // print output
    cout << "[ ";
    double *out = net->getOutput();
    int oNeurons = net->getOutputNeurons();
    
    cout << *out++;
    for (int i = 1; i<oNeurons; i++) {
        cout << ", " << *out;
        out++;
    }
    cout << " ]" << endl;
}

void printSeperator() {
    cout << endl;
    cout << "################################################################################" << endl;
    cout << endl;
}

void runTest(Network *net) {
    
    double input[] = {0, 0};
    net->setInput(input);
    net->run();
    printOutput(net);
    
    double input2[] = {0, 1};
    net->setInput(input2);
    net->run();
    printOutput(net);
    
    double input3[] = {1, 0};
    net->setInput(input3);
    net->run();
    printOutput(net);
    
    double input4[] = {1, 1};
    net->setInput(input4);
    net->run();
    printOutput(net);
    
}

LabeledDataset* createXorDataset() {
    SimpleLabeledDataset *ds = new SimpleLabeledDataset(2, 1, 4);
    
    ds->addPattern((const double[2]){0.0f, 0.0f}, (const double[1]){0.0f});
    ds->addPattern((const double[2]){0.0f, 1.0f}, (const double[1]){1.0f});
    ds->addPattern((const double[2]){1.0f, 0.0f}, (const double[1]){1.0f});
    ds->addPattern((const double[2]){1.0f, 1.0f}, (const double[1]){0.0f});
    
    return (LabeledDataset*)ds;
}

LabeledDataset* createAndDataset() {
    SimpleLabeledDataset *ds = new SimpleLabeledDataset(2, 1, 4);
    
    ds->addPattern((const double[2]){0.0f, 0.0f}, (const double[1]){0.0f});
    ds->addPattern((const double[2]){0.0f, 1.0f}, (const double[1]){0.0f});
    ds->addPattern((const double[2]){1.0f, 0.0f}, (const double[1]){0.0f});
    ds->addPattern((const double[2]){1.0f, 1.0f}, (const double[1]){1.0f});
    
    return (LabeledDataset*)ds;
}

// entry point of the application
int main(int argc, char *argv[]) {
    
    double targetMse = .0001;
    if (argc == 2) {
        targetMse = atoi(argv[1]);
    }
    
    NetworkConfiguration *conf = new NetworkConfiguration();

    conf->setLayers(3);
    conf->setNeurons(0, 2);
    conf->setNeurons(1, 2);
    conf->setNeurons(2, 1);
    conf->activationFnc = sigmoidFunction;
    conf->dActivationFnc = dSigmoidFunction;
    
    Network *net = new Network(conf);
    
    runTest(net);
    
    printSeperator();
    
    LabeledDataset *ds = createXorDataset();
    BackpropagationLearner *bp = new BackpropagationLearner(net);
    bp->setTargetMse(targetMse);
    bp->setErrorComputer(new MseErrorComputer());
    bp->train(ds);
    
    runTest(net);
    
    delete bp;
    delete ds;
    delete conf;
    delete net;
    return 0;
}

