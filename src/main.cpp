/* 
 * File:   main.cpp
 * Author: janvojt
 *
 * Created on May 29, 2014, 9:53 PM
 */

#include <cstdlib>
#include <iostream>
#include <getopt.h>

#include "NetworkConfiguration.h"
#include "Network.h"
#include "activationFunctions.h"
#include "BackpropagationLearner.h"
#include "SimpleLabeledDataset.h"
#include "MseErrorComputer.h"

// getopts constants
#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;

/* Application long options. */
const struct option optsLong[] = {
    {"help", no_argument, 0, 'h'},
    {"no-bias", no_argument, 0, 'b'},
    {"mse", required_argument, 0, 'e'},
    {"max-epochs", required_argument, 0, 'm'},
    {0, 0, 0, 0},
};

/* Application short options. */
const char* optsList = "hbe:m:";

/* Application configuration. */
struct config {
    /* mean square error */
    double mse = .0001;
    /* use bias? */
    bool bias = true;
    /* epoch limit */
    long maxEpochs = 1000000;
};


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

void printHelp() {
    cout << "Usage: xoraan [OPTIONS]" << endl << endl;
    cout << "Option      GNU long option       Meaning" << endl;
    cout << "-h          --help                This help." << endl;
    cout << "-b          --no-bias             Disables bias in neural network. Bias is enabled by default." << endl;
    cout << "-e <value>  --mse <value>         Target Mean Square Error to determine when to finish the learning." << endl;
    cout << "-m <value>  --max-epochs <value>  Sets a maximum limit for number of epochs. Learning is stopped even if MSE has not been met." << endl;
}

/* Process command line options and return generated configuration. */
config* processOptions(int argc, char *argv[]) {
    
    config* conf = new config;
    
    int index;
    int iarg = 0;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, optsList, optsLong, &index);

        switch (iarg) {
            case 'h':
                printHelp();
                exit(0);
            case 'b':
                conf->bias = false;
                break;
            case 'e':
                conf->mse = atof(optarg);
                break;
            case 'm' :
                conf->maxEpochs = atoi(optarg);
                break;
        }
    }
    
    return conf;
}

/* Entry point of the application. */
int main(int argc, char *argv[]) {
    
    config* conf = processOptions(argc, argv);
    
    NetworkConfiguration *netConf = new NetworkConfiguration();

    netConf->setLayers(3);
    netConf->setNeurons(0, 2);
    netConf->setNeurons(1, 2);
    netConf->setNeurons(2, 1);
    netConf->activationFnc = sigmoidFunction;
    netConf->dActivationFnc = dSigmoidFunction;
    netConf->setBias(conf->bias);
    
    Network *net = new Network(netConf);
    
    runTest(net);
    
    printSeperator();
    
    LabeledDataset *ds = createXorDataset();
    BackpropagationLearner *bp = new BackpropagationLearner(net);
    bp->setTargetMse(conf->mse);
    bp->setErrorComputer(new MseErrorComputer());
    bp->setEpochLimit(conf->maxEpochs);
    bp->train(ds);
    
    runTest(net);
    
    delete bp;
    delete ds;
    delete netConf;
    delete net;
    return 0;
}

