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
#include "ds/fold/FoldDatasetFactory.h"

// getopts constants
#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;

/* Maximum array size to print on stdout. */
const int MAX_PRINT_ARRAY_SIZE = 8;

/* Application short options. */
const char* optsList = "hbl:a:e:k:m:f:ic:s:t:v:q:r:ju:pd";

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
    {"lconf", required_argument, 0, 'c'},
    {"labels", required_argument, 0, 's'},
    {"test", required_argument, 0, 't'},
    {"validation", required_argument, 0, 'v'},
    {"k-fold", required_argument, 0, 'q'},
    {"idx", no_argument, 0, 'i'},
    {"random-seed", required_argument, 0, 'r'},
    {"shuffle", no_argument, 0, 'j'},
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
    double mse = .01;
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
    /* Determines whether to use the best fold in k-fold validation */
    bool useBestFold = false;
    /* Number of folds to be used in k-fold cross validation. */
    int kFold = 1;
    /* Use IDX data format when parsing input files? */
    bool useIdx = false;
    /* Seed for random generator. */
    int seed = 0;
    /* Determines whether training datasets should be shuffled. */
    bool shuffle = false;
    /* activation function to use */
    char actFunction = 's';
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

template <typename dType>
void printArray(dType *arr, int size) {
    
    // print input
    cout << "[ ";
    cout << arr[0];
    for (int i = 1; i<size; i++) {
        cout << ", " << arr[i];
    }
    cout << " ]";
}

template <typename dType>
void printInout(Network<dType> *net) {
    printArray(net->getInput(), net->getInputNeurons());
    cout << " -> ";
    printArray(net->getOutput(), net->getOutputNeurons());
    cout << endl;
}

/* Runs the given test dataset through given network and prints results. */
template <typename dType>
void runTest(Network<dType> *net, InputDataset<dType> *ds) {
    
    ds->reset();
    if (ds->getInputDimension() <= MAX_PRINT_ARRAY_SIZE) {
        while (ds->hasNext()) {
            net->setInput(ds->next());
            net->run();
            printInout(net);
        }
    } else {

        const SimpleLabeledDataset<dType> *LABELED_DATASET_CLASS = new SimpleLabeledDataset<dType>(0,0,0);
        LabeledDataset<dType> *lds = (LabeledDataset<dType> *) ds;
        ErrorComputer<dType> *ec = new MseErrorComputer<dType>();
        int i = 0;

        while (lds->hasNext()) {
            dType *pattern = lds->next();
            i++;
            dType *label = pattern + lds->getInputDimension();
            net->setInput(pattern);
            net->run();
            if (typeid(*ds)==typeid(*LABELED_DATASET_CLASS)) {
                dType error = ec->compute(net, label);
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

template <typename dType>
LabeledDataset<dType>* createXorDataset() {
    SimpleLabeledDataset<dType> *ds = new SimpleLabeledDataset<dType>(2, 1, 4);
    
    ds->addPattern((const dType[2]){0, 0}, (const dType[1]){0});
    ds->addPattern((const dType[2]){0, 1}, (const dType[1]){1});
    ds->addPattern((const dType[2]){1, 0}, (const dType[1]){1});
    ds->addPattern((const dType[2]){1, 1}, (const dType[1]){0});
    
    return (LabeledDataset<dType>*) ds;
}

void printHelp() {
    cout << "Usage: ffwdnet [OPTIONS]" << endl << endl;
    cout << "Option      GNU long option       Meaning" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "-h          --help                This help." << endl;
    cout << "-b          --no-bias             Disables bias in neural network. Bias is enabled by default." << endl;
    cout << "-l <value>  --rate <value>        Learning rate influencing the speed and quality of learning. Default value is 0.3." << endl;
    cout << "-a <value>  --init <value>        Minimum and maximum value network weights are initialized to. Default is -1,1." << endl;
    cout << "-e <value>  --mse <value>         Target Mean Square Error to determine when to finish the learning. Default is 0.01." << endl;
    cout << "-k <value>  --improve-err <value> Number of epochs during which improvement of error is required to keep learning. Default is zero (=disabled)." << endl;
    cout << "-m <value>  --max-epochs <value>  Sets a maximum limit for number of epochs. Learning is stopped even if MSE has not been met. Default is 100,000" << endl;
    cout << "-f <value>  --func <value>        Specifies the activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << "-c <value>  --lconf <value>       Specifies layer configuration for the network as a comma separated list of integers." << endl;
    cout << "-s <value>  --labels <value>      File path with labeled data to be used for learning." << endl;
    cout << "-t <value>  --test <value>        File path with test data to be used for evaluating networks performance." << endl;
    cout << "-v <value>  --validation <value>  Size of the validation set. Patterns are taken from the training set. Default is zero." << endl;
    cout << "-q <value>  --k-fold <value>      Number of folds to use in k-fold cross validation. Default is one (=disabled)." << endl;
    cout << "-o          --best-fold           Uses the best network trained with k-fold validation. By default epoch limit is averaged and network is trained on all data." << endl;
    cout << "-i          --idx                 Use IDX data format when parsing files with datasets. Human readable CSV-like format is the default." << endl;
    cout << "-r <value>  --random-seed <value> Specifies value to be used for seeding random generator." << endl;
    cout << "-j          --shuffle             Shuffles training and validation dataset do the patterns are in random order." << endl;
    cout << "-u <value>  --use-cache <value>   Enables use of precomputed lookup table for activation function. Value specifies the size of the table." << endl;
    cout << "-p          --use-gpu             Enables parallel implementation of the network using CUDA GPU API." << endl;
    cout << "-d          --debug               Enable debugging messages." << endl;
}

/* Process command line options and return generated configuration. */
config* processOptions(int argc, char *argv[]) {
    
    config* conf = new config;
    
    // set defaults
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
                conf->kFold = 0;
                break;
            case 'q' :
                conf->validationSize = 0;
                conf->kFold = atoi(optarg);
                break;
            case 'o' :
                conf->useBestFold = true;
                break;
            case 'i' :
                conf->useIdx = true;
                break;
            case 'r' :
                conf->seed = atoi(optarg);
                break;
            case 'j' :
                conf->shuffle = true;
                break;
            case 'f' :
                switch (optarg[0]) {
                    case 's' :
                        conf->actFunction = 's';
                        break;
                    case 'h' :
                        conf->actFunction = 'h';
                        break;
                    default :
                        LOG()->warn("Unknown activation function %s, falling back to sigmoid.", optarg);
                        conf->actFunction = 's';
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
template <typename dType>
NetworkConfiguration<dType> *createNetworkConfiguration(config* conf) {
    
    LOG()->info("Seeding random generator with %d.", conf->seed);
    srand(conf->seed);
    
    NetworkConfiguration<dType> *netConf = new NetworkConfiguration<dType>();
    
    // Setup function lookup table if it is to be used.
    if (conf->useGpu && conf->useFunctionCache) {
        LOG()->warn("Precomputed activation function table is only supported on CPU, disabling.");
    } else if (conf->useFunctionCache) {
        switch (conf->actFunction) {
            case 's' :
                FunctionCache<DATA_TYPE>::init(sigmoidFunction, conf->functionSamples);
                netConf->dActivationFnc = dSigmoidFunction;
                break;
            case 'h' :
                FunctionCache<DATA_TYPE>::init(hyperbolicTangentFunction, conf->functionSamples);
                netConf->dActivationFnc = dHyperbolicTangentFunction;
                break;
        }
        netConf->activationFnc = cachedFunction;
    } else {
        switch (conf->actFunction) {
            case 's' :
                LOG()->info("Using sigmoid as activation function.");
                netConf->activationFnc = sigmoidFunction;
                netConf->dActivationFnc = dSigmoidFunction;
                break;
            case 'h' :
                LOG()->info("Using hyperbolic tangent as activation function.");
                netConf->activationFnc = hyperbolicTangentFunction;
                netConf->dActivationFnc = dHyperbolicTangentFunction;
                break;
        }
    }
    
    // Setup network configuration.
    netConf->parseLayerConf(conf->layerConf);
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

template <typename dType>
void printImageLabels(LabeledDataset<dType> *lds) {
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
    
    LOG()->info("Compiled with %dbit precision for data types.", 8*sizeof(DATA_TYPE));
        
    // prepare network configuration
    config* conf = processOptions(argc, argv);

    // Configure seed for random generator.
    if (conf->seed == 0) {
        conf->seed = getSeed();
    }
    
    NetworkConfiguration<DATA_TYPE> *netConf = createNetworkConfiguration<DATA_TYPE>(conf);
    
    // construct the network
    Network<DATA_TYPE> *net;
    BackpropagationLearner<DATA_TYPE> *bp;
    
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
        GpuNetwork<DATA_TYPE> *gpuNet = new GpuNetwork<DATA_TYPE>(netConf, gpuConf);
        bp = new GpuBackpropagationLearner<DATA_TYPE>(gpuNet);
        net = gpuNet;
    } else {
        LOG()->info("Using CPU for computing the network runs.");
        CpuNetwork<DATA_TYPE> *cpuNet = new CpuNetwork<DATA_TYPE>(netConf);
        bp = new CpuBackpropagationLearner<DATA_TYPE>(cpuNet);
        net = cpuNet;
    }
    
    // Prepare test dataset.
    InputDataset<DATA_TYPE> *tds;
    if (conf->testData == NULL) {
        tds = createXorDataset<DATA_TYPE>();
    } else if (conf->useIdx) {
        LabeledMnistParser<DATA_TYPE> *p = new LabeledMnistParser<DATA_TYPE>();
        tds = p->parse(conf->testData);
//        printImageLabels((LabeledDataset *)tds);
//        return 0;
        delete p;
    } else {
        InputDatasetParser<DATA_TYPE> *p = new InputDatasetParser<DATA_TYPE>(conf->testData, netConf);
        tds = p->parse();
        delete p;
    }
    
    // Prepare training dataset.
    // If none was provided in options use XOR dataset by default.
    LabeledDataset<DATA_TYPE> *lds;
    if (conf->labeledData == NULL) {
        lds = createXorDataset<DATA_TYPE>();
    } else if (conf->useIdx) {
        LabeledMnistParser<DATA_TYPE> *p = new LabeledMnistParser<DATA_TYPE>();
        lds = p->parse(conf->labeledData);
        delete p;
    } else {
        LabeledDatasetParser<DATA_TYPE> *p = new LabeledDatasetParser<DATA_TYPE>(conf->labeledData, netConf);
        lds = p->parse();
        delete p;
    }
    
    // Shuffle labeled data
    if (conf->shuffle) {
        LOG()->info("Randomly shuffling the training dataset.");
        lds->shuffle();
    }
    
//    printImageLabels((LabeledDataset *)lds);
//    return 0;
    
    // configure BP learner
    bp->setLearningRate(conf->lr);
    bp->setTargetMse(conf->mse);
    bp->setErrorComputer(new MseErrorComputer<DATA_TYPE>());
    bp->setEpochLimit(conf->maxEpochs);
    bp->setImproveEpochs(conf->improveEpochs);
    
    // Prepare validation dataset
    FoldDatasetFactory<DATA_TYPE> *df;
    LabeledDataset<DATA_TYPE> *vds;
    if (conf->kFold > 1) {
        
        LOG()->info("Training with %d-fold cross validation.", conf->kFold);
        df = new FoldDatasetFactory<DATA_TYPE>(lds, conf->kFold);
        DATA_TYPE bestError = 2.0; // start with impossibly bad error
        DATA_TYPE bestValIdx = 0;
        Network<DATA_TYPE> *bestNet = NULL;
        long totalEpochs = 0;
        
        // train each fold
        for (int i = 0; i<conf->kFold; i++) {
            LabeledDataset<DATA_TYPE> *t = (LabeledDataset<DATA_TYPE> *) df->getTrainingDataset(i);
            LabeledDataset<DATA_TYPE> *v = (LabeledDataset<DATA_TYPE> *) df->getValidationDataset(i);
            
            net->reinit();
            TrainingResult<DATA_TYPE> *res = bp->train(t, v, 0);
            totalEpochs += res->getEpochs();
            
            if (conf->useBestFold && res->getValidationError() < bestError) {
                if (bestNet != NULL) delete bestNet;
                bestError = res->getValidationError();
                bestValIdx = i;
                bestNet = net->clone();
            }
            
            delete t;
            delete v;
            delete res;
        }
        
        // use the best network
        if (conf->useBestFold) {
            LOG()->info("Best error of %f achieved using validation fold %d.", bestError, bestValIdx);
            delete net;
            net = bestNet;
        } else {
            bp->setEpochLimit(totalEpochs / conf->kFold);
            bp->train(lds, vds, 0);
        }

    } else if (conf->validationSize > 0) {
        
        LOG()->info("Picking the last %d samples from the training dataset to be used for validation.");
        vds = lds->takeAway(conf->validationSize);
        bp->train(lds, vds, 0);
        
    } else {
        
        vds = (LabeledDataset<DATA_TYPE> *) new SimpleLabeledDataset<DATA_TYPE>(0, 0, 0);
        bp->train(lds, vds, 0);
    }
    
    // Run (hopefully) learnt network.
    runTest<DATA_TYPE>(net, tds);
    
    // Release dynamically allocated memory
    if (conf->kFold > 1) {
        delete df;
    }
    
    delete bp;
    delete lds;
    delete tds;
    delete netConf;
    delete conf;
    delete net;
    return 0;
}

