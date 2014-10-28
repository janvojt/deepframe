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
#include <unistd.h>

#include "NetworkConfiguration.h"
#include "Network.h"
#include "activationFunctions.h"
#include "BackpropagationLearner.h"
#include "SimpleLabeledDataset.h"
#include "MseErrorComputer.h"
#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"
#include "log4cpp/Priority.hh"
#include "LabeledDatasetParser.h"
#include "SimpleInputDataset.h"
#include "InputDatasetParser.h"

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
    {"labels", required_argument, 0, 's'},
    {"test", required_argument, 0, 't'},
    {"random-seed", required_argument, 0, 'r'},
    {"debug", no_argument, 0, 'd'},
    {0, 0, 0, 0},
};

/* Application short options. */
const char* optsList = "hbe:m:f:g:i:l:s:t:r:d";

/* Application configuration. */
struct config {
    /* mean square error */
    double mse = .0001;
    /* use bias? */
    bool bias = true;
    /* epoch limit */
    long maxEpochs = 100000;
    /* Layer configuration. */
    char* layerConf = NULL;
    /* File path with labeled data. */
    char* labeledData = NULL;
    /* File path with test data. */
    char* testData = NULL;
    /* Seed for random generator. */
    int seed = 0;
    /* activation function */
    void (*activationFnc)(double *x, double *y, int layerSize);
    /* derivative of activation function */
    void (*dActivationFnc)(double *x, double *y, int layerSize);
};

/* Random seed generator. */
unsigned getSeed(void)
{
  unsigned seed = 0;
  
  int fp = open("/dev/urandom", O_RDONLY);
  read(fp, &seed, sizeof seed);
  close(fp);
  
  return seed;
}


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

/* Runs the given test dataset through given network and prints results. */
void runTest(Network *net, InputDataset *ds) {
    ds->reset();
    while (ds->hasNext()) {
        net->setInput(ds->next());
        net->run();
        printOutput(net);
    }
}

LabeledDataset* createXorDataset() {
    SimpleLabeledDataset *ds = new SimpleLabeledDataset(2, 1, 4);
    
    ds->addPattern((const double[2]){0, 0}, (const double[1]){0});
    ds->addPattern((const double[2]){0, 1}, (const double[1]){1});
    ds->addPattern((const double[2]){1, 0}, (const double[1]){1});
    ds->addPattern((const double[2]){1, 1}, (const double[1]){0});
    
    return (LabeledDataset*)ds;
}

void printHelp() {
    cout << "Usage: xoraan [OPTIONS]" << endl << endl;
    cout << "Option      GNU long option       Meaning" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "-h          --help                This help." << endl;
    cout << "-b          --no-bias             Disables bias in neural network. Bias is enabled by default." << endl;
    cout << "-e <value>  --mse <value>         Target Mean Square Error to determine when to finish the learning." << endl;
    cout << "-m <value>  --max-epochs <value>  Sets a maximum limit for number of epochs. Learning is stopped even if MSE has not been met. Default is 100,000" << endl;
    cout << "-f <value>  --func <value>        Specifies the activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-g <value>  --d-func <value>      Specifies the derivative of activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-l <value>  --lconf <value>       Specifies layer configuration for the network as a comma separated list of integers." << endl;
    cout << "-s <value>  --labels <value>      File path with labeled data to be used for learning." << endl;
    cout << "-t <value>  --test <value>        File path with test data to be used for evaluating networks performance." << endl;
    cout << "-r <value>  --random-seed <value> Specifies value to be used for seeding random generator." << endl;
    cout << "-d          --debug               Enable debugging messages." << endl;
}

/* Process command line options and return generated configuration. */
config* processOptions(int argc, char *argv[]) {
    
    config* conf = new config;
    
    // set defaults
    conf->activationFnc = sigmoidFunction;
    conf->dActivationFnc = dSigmoidFunction;
    conf->layerConf = (char*) "2,2,1";
    
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
                conf->maxEpochs = atol(optarg);
                break;
            case 'l' :
                conf->layerConf = new char[strlen(optarg)+1];
                strcpy(conf->layerConf, optarg);
                break;
            case 's' :
                conf->labeledData = optarg;
                break;
            case 't' :
                conf->testData = optarg;
                break;
            case 'r' :
                conf->seed = atoi(optarg);
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
    
    // Seed random generator before initializing weights.
    if (conf->seed == 0) {
        conf->seed = getSeed();
    }
    LOG()->info("Seeding random generator with %d.", conf->seed);
    srand(conf->seed);
    
    // Setup network configuration.
    NetworkConfiguration *netConf = new NetworkConfiguration();
    netConf->parseLayerConf(conf->layerConf);
    netConf->activationFnc = conf->activationFnc;
    netConf->dActivationFnc = conf->dActivationFnc;
    netConf->setBias(conf->bias);
    
    Network *net = new Network(netConf);
    
    // Prepare test dataset.
    InputDataset *tds;
    if (conf->testData == NULL) {
        SimpleInputDataset *d = new SimpleInputDataset(2, 4);
        d->addInput((const double[2]){0, 0});
        d->addInput((const double[2]){0, 1});
        d->addInput((const double[2]){1, 0});
        d->addInput((const double[2]){1, 1});
        tds = (InputDataset *) d;
    } else {
        InputDatasetParser *p = new InputDatasetParser(conf->testData, netConf);
        tds = p->parse();
        delete p;
    }
    
    // Run network without learning.
    runTest(net, tds);
    printSeperator();

    // Prepare labeled dataset.
    // If none was provided in options use XOR dataset by default.
    LabeledDataset *ds;
    if (conf->labeledData == NULL) {
        ds = createXorDataset();
    } else {
        LabeledDatasetParser *p = new LabeledDatasetParser(conf->labeledData, netConf);
        ds = p->parse();
        delete p;
    }
    
    BackpropagationLearner *bp = new BackpropagationLearner(net);
    bp->setTargetMse(conf->mse);
    bp->setErrorComputer(new MseErrorComputer());
    bp->setEpochLimit(conf->maxEpochs);
    bp->train(ds);
    
    // Run (hopefully) learnt network.
    runTest(net, tds);
    
    delete bp;
    delete ds;
    delete tds;
    delete netConf;
    delete conf;
    delete net;
    return 0;
}

