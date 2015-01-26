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
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <typeinfo>

#include "net/NetworkConfiguration.h"
#include "net/Network.h"
#include "net/CpuNetwork.h"
#include "net/GpuNetwork.h"
#include "net/CpuNetwork.h"
#include "net/GpuConfiguration.h"
#include "ds/SimpleLabeledDataset.h"
#include "ds/LabeledDatasetParser.h"
#include "ds/SimpleInputDataset.h"
#include "ds/InputDatasetParser.h"
#include "ds/LabeledMnistParser.h"
#include "bp/BackpropagationLearner.h"
#include "bp/CpuBackpropagationLearner.h"
#include "bp/GpuBackpropagationLearner.h"
#include "err/MseErrorComputer.h"
#include "activationFunctions.h"
#include "FunctionCache.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"
#include "log4cpp/Priority.hh"

// getopts constants
#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;

/* Maximum array size to print on stdout. */
const int MAX_PRINT_ARRAY_SIZE = 8;

/* Application short options. */
const char* optsList = "hbl:a:e:k:m:f:g:ic:s:t:v:r:u:pd";

/* Application long options. */
const struct option optsLong[] = {
    {"help", no_argument, 0, 'h'},
    {"no-bias", no_argument, 0, 'b'},
    {"rate", required_argument, 0, 'l'},
    {"init", required_argument, 0, 'a'},
    {"mse", required_argument, 0, 'e'},
    {"max-epochs", required_argument, 0, 'm'},
    {"improve-err", required_argument, 0, 'k'},
    {"func", required_argument, 0, 'f'},
    {"d-func", required_argument, 0, 'g'},
    {"lconf", required_argument, 0, 'c'},
    {"labels", required_argument, 0, 's'},
    {"test", required_argument, 0, 't'},
    {"validation", required_argument, 0, 'v'},
    {"idx", no_argument, 0, 'i'},
    {"random-seed", required_argument, 0, 'r'},
    {"use-cache", optional_argument, 0, 'u'},
    {"use-gpu", no_argument, 0, 'p'},
    {"debug", no_argument, 0, 'd'},
    {0, 0, 0, 0},
};

/* Application configuration. */
struct config {
    /* learning rate */
    double lr = .3;
    /* Interval network weights are initialized in. */
    char* initInterval = NULL;
    /* mean square error */
    double mse = .0001;
    /* Number of epochs during which error improvement
     *  is required to keep learning. */
    int improveEpochs = 0;
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
    /* Validation data set size */
    int validationSize = 0;
    /* Use IDX data format when parsing input files? */
    bool useIdx = false;
    /* Seed for random generator. */
    int seed = 0;
    /* activation function */
    void (*activationFnc)(double *x, double *y, int layerSize);
    /* derivative of activation function */
    void (*dActivationFnc)(double *x, double *y, int layerSize);
    /* Whether to use precomputed table for activation function value lookup. */
    bool useFunctionCache = false;
    /* Number of samples for function cache lookup table. */
    int functionSamples = 10000;
    /* Whether to use parallel implementation using GPU. */
    bool useGpu = false;
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

void printArray(double *arr, int size) {
    
    // print input
    cout << "[ ";
    cout << arr[0];
    for (int i = 1; i<size; i++) {
        cout << ", " << arr[i];
    }
    cout << " ]";
}

void printInout(Network *net) {
    printArray(net->getInput(), net->getInputNeurons());
    cout << " -> ";
    printArray(net->getOutput(), net->getOutputNeurons());
    cout << endl;
}

const SimpleLabeledDataset *LABELED_DATASET_CLASS = new SimpleLabeledDataset(0,0,0);

/* Runs the given test dataset through given network and prints results. */
void runTest(Network *net, InputDataset *ds) {
    
    ds->reset();
    if (ds->getInputDimension() <= MAX_PRINT_ARRAY_SIZE) {
        while (ds->hasNext()) {
            net->setInput(ds->next());
            net->run();
            printInout(net);
        }
    } else {
        LabeledDataset *lds = (LabeledDataset *) ds;
        ErrorComputer *ec = new MseErrorComputer();
        int i = 0;
        while (lds->hasNext()) {
            double *pattern = lds->next();
            i++;
            double *label = pattern + lds->getInputDimension();
            net->setInput(pattern);
            net->run();
            if (typeid(*ds)==typeid(*LABELED_DATASET_CLASS)) {
                double error = ec->compute(net, label);
                cout << "Output for pattern " << i << ": ";
                printArray(net->getOutput(), net->getOutputNeurons());
                cout << ", label: ";
                printArray(label, net->getOutputNeurons());
                cout << ", MSE error: " << error << endl;
            } else {
                cout << "Output for pattern " << i << ": ";
                printArray(net->getOutput(), net->getOutputNeurons());
                cout << "." << endl;
            }
        }
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
    cout << "Usage: ffwdnet [OPTIONS]" << endl << endl;
    cout << "Option      GNU long option       Meaning" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "-h          --help                This help." << endl;
    cout << "-b          --no-bias             Disables bias in neural network. Bias is enabled by default." << endl;
    cout << "-l <value>  --rate <value>        Learning rate influencing the speed and quality of learning. Default value is 0.3." << endl;
    cout << "-a <value>  --init <value>        Minimum and maximum value network weights are initialized to. Default is -1,1." << endl;
    cout << "-e <value>  --mse <value>         Target Mean Square Error to determine when to finish the learning." << endl;
    cout << "-k <value>  --improve-err <value> Number of epochs during which improvement of error is required to keep learning. Default is zero (=disabled)." << endl;
    cout << "-m <value>  --max-epochs <value>  Sets a maximum limit for number of epochs. Learning is stopped even if MSE has not been met. Default is 100,000" << endl;
    cout << "-f <value>  --func <value>        Specifies the activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-g <value>  --d-func <value>      Specifies the derivative of activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-c <value>  --lconf <value>       Specifies layer configuration for the network as a comma separated list of integers." << endl;
    cout << "-s <value>  --labels <value>      File path with labeled data to be used for learning." << endl;
    cout << "-t <value>  --test <value>        File path with test data to be used for evaluating networks performance." << endl;
    cout << "-v <value>  --validation <value>  Size of the validation set. Patterns are taken from the training set. Default is zero." << endl;
    cout << "-i          --idx                 Use IDX data format when parsing files with datasets. Human readable CSV-like format is the default." << endl;
    cout << "-r <value>  --random-seed <value> Specifies value to be used for seeding random generator." << endl;
    cout << "-u <value>  --use-cache <value>   Enables use of precomputed lookup table for activation function. Value specifies the size of the table." << endl;
    cout << "-p          --use-gpu             Enables parallel implementation of the network using CUDA GPU API." << endl;
    cout << "-d          --debug               Enable debugging messages." << endl;
}

/* Process command line options and return generated configuration. */
config* processOptions(int argc, char *argv[]) {
    
    config* conf = new config;
    
    // set defaults
    conf->activationFnc = sigmoidFunction;
    conf->dActivationFnc = dSigmoidFunction;
    conf->layerConf = (char*) "2,2,1";
    conf->initInterval = (char*) "-1,1";
    
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
            case 'a':
                conf->initInterval = new char[strlen(optarg)+1];
                strcpy(conf->initInterval, optarg);
                break;
            case 'l':
                conf->lr = atof(optarg);
                break;
            case 'e':
                conf->mse = atof(optarg);
                break;
            case 'k':
                conf->improveEpochs = atoi(optarg);
                break;
            case 'm' :
                conf->maxEpochs = atol(optarg);
                break;
            case 'c' :
                conf->layerConf = new char[strlen(optarg)+1];
                strcpy(conf->layerConf, optarg);
                break;
            case 's' :
                conf->labeledData = optarg;
                break;
            case 't' :
                conf->testData = optarg;
                break;
            case 'v' :
                conf->validationSize = atoi(optarg);
                break;
            case 'i' :
                conf->useIdx = true;
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
            case 'u' :
                conf->useFunctionCache = true;
                if (optarg != NULL) {
                    conf->functionSamples = atoi(optarg);
                }
                break;
            case 'p' :
                conf->useGpu = true;
                break;
        }
    }
    
    return conf;
}

/* Factory for network configuration. */
NetworkConfiguration *createNetworkConfiguration(config* conf) {
    
    LOG()->info("Seeding random generator with %d.", conf->seed);
    srand(conf->seed);
    
    // Setup function lookup table if it is to be used.
    if (conf->useFunctionCache) {
        FunctionCache::init(conf->activationFnc, conf->functionSamples);
        conf->activationFnc = cachedFunction;
    }
    
    // Setup network configuration.
    NetworkConfiguration *netConf = new NetworkConfiguration();
    netConf->parseLayerConf(conf->layerConf);
    netConf->activationFnc = conf->activationFnc;
    netConf->dActivationFnc = conf->dActivationFnc;
    netConf->setBias(conf->bias);
    netConf->parseInitInterval(conf->initInterval);
    
    return netConf;
}

/* Factory for GPU configuration. */
GpuConfiguration *createGpuConfiguration(config *conf) {
    
    GpuConfiguration *gpuConf = GpuConfiguration::create();
    if (gpuConf == NULL) return NULL;
    
    curandGenerator_t *gen = new curandGenerator_t;
    curandCreateGenerator(gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(*gen, conf->seed);
    gpuConf->setRandGen(gen);
    
    return gpuConf;
}

void printImage(int x, int y, double *arr) {
    for (int i = 0; i<x; i++) {
        char sep = ' ';
        cout << sep;
        for (int j = 0; j<y; j++) {
            double d = arr[i*x+j];
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
        double* x = lds->next();
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

/* Entry point of the application. */
int main(int argc, char *argv[]) {
        
    // prepare network configuration
    config* conf = processOptions(argc, argv);

    // Configure seed for random generator.
    if (conf->seed == 0) {
        conf->seed = getSeed();
    }
    
    NetworkConfiguration *netConf = createNetworkConfiguration(conf);
    
    // construct the network
    Network *net;
    BackpropagationLearner *bp;
    
    // probe GPU and fetch specs
    GpuConfiguration *gpuConf;
    bool useGpu = conf->useGpu;
    if (useGpu) {
        gpuConf = createGpuConfiguration(conf);
        if (gpuConf == NULL) {
            LOG()->warn("Falling back to CPU as GPU probe was unsuccessful.");
            useGpu = false;
        }
    }
    
    // setup correct implementations for network and BP learner
    if (useGpu) {
        LOG()->info("Using GPU for computing the network runs.");
        GpuNetwork *gpuNet = new GpuNetwork(netConf, gpuConf);
        bp = new GpuBackpropagationLearner(gpuNet);
        net = gpuNet;
    } else {
        LOG()->info("Using CPU for computing the network runs.");
        CpuNetwork *cpuNet = new CpuNetwork(netConf);
        bp = new CpuBackpropagationLearner(cpuNet);
        net = cpuNet;
    }
    
    // Prepare test dataset.
    InputDataset *tds;
    if (conf->testData == NULL) {
        SimpleInputDataset *d = new SimpleInputDataset(2, 4);
        d->addInput((const double[2]){0, 0});
        d->addInput((const double[2]){0, 1});
        d->addInput((const double[2]){1, 0});
        d->addInput((const double[2]){1, 1});
        tds = (InputDataset *) d;
    } else if (conf->useIdx) {
        LabeledMnistParser *p = new LabeledMnistParser();
        tds = p->parse(conf->testData);
//        printImageLabels((LabeledDataset *)tds);
//        return 0;
        delete p;
    } else {
        InputDatasetParser *p = new InputDatasetParser(conf->testData, netConf);
        tds = p->parse();
        delete p;
    }
    
    // Prepare labeled dataset.
    // If none was provided in options use XOR dataset by default.
    LabeledDataset *lds;
    if (conf->labeledData == NULL) {
        lds = createXorDataset();
    } else if (conf->useIdx) {
        LabeledMnistParser *p = new LabeledMnistParser();
        lds = p->parse(conf->labeledData);
//        printImageLabels((LabeledDataset *)lds);
//        return 0;
        delete p;
    } else {
        LabeledDatasetParser *p = new LabeledDatasetParser(conf->labeledData, netConf);
        lds = p->parse();
        delete p;
    }
    
    // Prepare validation dataset
    LabeledDataset *vds;
    if (conf->validationSize > 0) {
        vds = lds->takeAway(conf->validationSize);
    } else {
        vds = (LabeledDataset *) new SimpleLabeledDataset(0, 0, 0);
    }
    
    // configure BP learner
    bp->setLearningRate(conf->lr);
    bp->setTargetMse(conf->mse);
    bp->setErrorComputer(new MseErrorComputer());
    bp->setEpochLimit(conf->maxEpochs);
    bp->setImproveEpochs(conf->improveEpochs);
    
    // train network
    bp->train(lds, vds);
    
    // Run (hopefully) learnt network.
    runTest(net, tds);
    
    delete bp;
    delete lds;
    delete tds;
    delete netConf;
    delete conf;
    delete net;
    return 0;
}

