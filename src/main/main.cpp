/* 
 * File:   main.cpp
 * Author: janvojt
 *
 * Created on May 29, 2014, 9:53 PM
 */

#include <cstdlib>
#include <string.h>
#include <iostream>
#include <getopt.h>

#include "NetworkConfiguration.h"
#include "Network.h"
#include "activationFunctions.h"
#include "BackpropagationLearner.h"
#include "SimpleLabeledDataset.h"
#include "MseErrorComputer.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"
#include "log4cpp/Priority.hh"

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
    {"func", required_argument, 0, 'f'},
    {"d-func", required_argument, 0, 'g'},
    {"init", required_argument, 0, 'i'},
    {"lconf", required_argument, 0, 'l'},
    {"debug", no_argument, 0, 'd'},
    {0, 0, 0, 0},
};

/* Application short options. */
const char* optsList = "hbe:m:f:g:i:l:d";

/* Application configuration. */
struct config {
    /* mean square error */
    double mse = .0001;
    /* use bias? */
    bool bias = true;
    /* epoch limit */
    long maxEpochs = 1000000;
    /* Weights and bias initialization. */
    bool initRandom = true;
    double initWeights = 0;
    /* Layer configuration. */
    char* layerConf;
    /* activation function */
    void (*activationFnc)(double *x, double *y, int layerSize);
    /* derivative of activation function */
    void (*dActivationFnc)(double *x, double *y, int layerSize);
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
    
    ds->addPattern((const double[2]){0, 0}, (const double[1]){0});
    ds->addPattern((const double[2]){0, 1}, (const double[1]){1});
    ds->addPattern((const double[2]){1, 0}, (const double[1]){1});
    ds->addPattern((const double[2]){1, 1}, (const double[1]){0});
    
    return (LabeledDataset*)ds;
}

LabeledDataset* createAndDataset() {
    SimpleLabeledDataset *ds = new SimpleLabeledDataset(2, 1, 4);
    
    ds->addPattern((const double[2]){0, 0}, (const double[1]){0});
    ds->addPattern((const double[2]){0, 1}, (const double[1]){0});
    ds->addPattern((const double[2]){1, 0}, (const double[1]){0});
    ds->addPattern((const double[2]){1, 1}, (const double[1]){1});
    
    return (LabeledDataset*)ds;
}

void printHelp() {
    cout << "Usage: xoraan [OPTIONS]" << endl << endl;
    cout << "Option      GNU long option       Meaning" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "-h          --help                This help." << endl;
    cout << "-b          --no-bias             Disables bias in neural network. Bias is enabled by default." << endl;
    cout << "-e <value>  --mse <value>         Target Mean Square Error to determine when to finish the learning." << endl;
    cout << "-m <value>  --max-epochs <value>  Sets a maximum limit for number of epochs. Learning is stopped even if MSE has not been met." << endl;
    cout << "-f <value>  --func <value>        Specifies the activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-g <value>  --d-func <value>      Specifies the derivative of activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-i <value>  --init <value>        Specifies the value all weights and biases should be initialized to. By default random initialization is used." << endl;
    cout << "-l <value>  --lconf <value>       Specifies layer configuration for the network as a comma separated list of integers." << endl;
    cout << "-d          --debug               Enable debugging messages." << endl;
}

/* Process command line options and return generated configuration. */
config* processOptions(int argc, char *argv[]) {
    
    config* conf = new config;
    
    // set defaults
    conf->activationFnc = sigmoidFunction;
    conf->dActivationFnc = dSigmoidFunction;
    conf->layerConf = "2,2,1";
    
    int index;
    int iarg = 0;
    while (iarg != -1) {
        iarg = getopt_long(argc, argv, optsList, optsLong, &index);

        switch (iarg) {
            case 'h':
                printHelp();
                exit(0);
            case 'd' :
                LOG()->setPriority(log4cpp::Priority::DEBUG);
                break;
            case 'b':
                conf->bias = false;
                break;
            case 'e':
                conf->mse = atof(optarg);
                break;
            case 'm' :
                conf->maxEpochs = atoi(optarg);
                break;
            case 'i' :
                conf->initWeights = atof(optarg);
                conf->initRandom = false;
            case 'l' :
                conf->layerConf = new char[strlen(optarg)];
                strcpy(conf->layerConf, optarg);
                break;
            case 'f' :
                switch (optarg[0]) {
                    case 's' :
                        conf->activationFnc = sigmoidFunction;
                        break;
                    case 'h' :
                        conf->activationFnc = hyperbolicTangentFunction;
                        break;
                    default :
                        LOG()->warn("Unknown activation function %s, falling back to sigmoid.", optarg);
                        conf->activationFnc = sigmoidFunction;
                        break;
                }
                break;
            case 'g' :
                switch (optarg[0]) {
                    case 's' :
                        conf->dActivationFnc = dSigmoidFunction;
                        break;
                    case 'h' :
                        conf->dActivationFnc = dHyperbolicTangentFunction;
                        break;
                    default :
                        LOG()->warn("Unknown derivative of activation function %s, falling back to sigmoid derivative.", optarg);
                        conf->dActivationFnc = dSigmoidFunction;
                        break;
                }
                break;
        }
    }
    
    return conf;
}

/* Entry point of the application. */
int main(int argc, char *argv[]) {
    
    config* conf = processOptions(argc, argv);
    
    NetworkConfiguration *netConf = new NetworkConfiguration();

    // Configure layers.
    // Count and set number of layers.
    int i;
    char *lconf = conf->layerConf;
    for (i=0; lconf[i]; lconf[i]==',' ? i++ : *lconf++);
    netConf->setLayers(i+1);
    
    // set number of neurons for each layer
    i = 0;
    int l = 0;
    char *haystack = new char[strlen(conf->layerConf)];
    strcpy(haystack, conf->layerConf);
    char *token = strtok(haystack, ",");
    while (token != NULL) {
        sscanf(token, "%d", &l);
        netConf->setNeurons(i++, l);
        token = strtok(NULL, ",");
    }
    
    // configure other network properties
    netConf->activationFnc = conf->activationFnc;
    netConf->dActivationFnc = conf->dActivationFnc;
    netConf->setBias(conf->bias);
    netConf->setInitRandom(conf->initRandom);
    netConf->setInitWeights(conf->initWeights);
    
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

