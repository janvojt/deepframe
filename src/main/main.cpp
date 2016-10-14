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
#include "net/NetworkSerializer.h"
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
#include "train/BackpropagationLearner.h"
#include "train/NetworkPretrainer.h"
#include "err/MseErrorComputer.h"
#include "activationFunctions.h"
#include "FunctionCache.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"
#include "log4cpp/Priority.hh"
#include "ds/fold/FoldDatasetFactory.h"
#include "net/layers/SubsamplingLayer.h"
#include "net/layers/FullyConnectedLayer.h"

#include "util/cpuDebugHelpers.h"
#include "ds/float/FloatDataset.h"

// getopts constants
#define no_argument 0
#define required_argument 1 
#define optional_argument 2

using namespace std;

/* Maximum array size to print on stdout. */
const int MAX_PRINT_ARRAY_SIZE = 8;

/* Application short options. */
const char* optsList = "hbl:a:e:k:m:n:f:igc:s:t:v:q:r:w:x:ju:pd";

/* Application long options. */
const struct option optsLong[] = {
    {"help", no_argument, 0, 'h'},
    {"no-bias", no_argument, 0, 'b'},
    {"rate", required_argument, 0, 'l'},
    {"init", required_argument, 0, 'a'},
    {"mse", required_argument, 0, 'e'},
    {"max-epochs", required_argument, 0, 'm'},
    {"pretrain", required_argument, 0, 'n'},
    {"improve-err", required_argument, 0, 'k'},
    {"func", required_argument, 0, 'f'},
    {"lconf", required_argument, 0, 'c'},
    {"labels", required_argument, 0, 's'},
    {"test", required_argument, 0, 't'},
    {"validation", required_argument, 0, 'v'},
    {"k-fold", required_argument, 0, 'q'},
    {"idx", no_argument, 0, 'i'},
    {"float", no_argument, 0, 'g'},
    {"random-seed", required_argument, 0, 'r'},
    {"export", required_argument, 0, 'w'},
    {"import", required_argument, 0, 'x'},
    {"shuffle", no_argument, 0, 'j'},
    {"use-cache", optional_argument, 0, 'u'},
    {"use-gpu", no_argument, 0, 'p'},
    {"debug", no_argument, 0, 'd'},
    {0, 0, 0, 0},
};

/* Application configuration. */
struct config {
    /* learning rate */
    data_t lr = .3;
    /* Interval network weights are initialized in. */
    char* initInterval = NULL;
    /* mean square error */
    data_t mse = .01;
    /* Number of epochs during which error improvement
     *  is required to keep learning. */
    int improveEpochs = 0;
    /* use bias? */
    bool bias = true;
    /* epoch limit */
    long maxEpochs = 100000;
    /* number of pretraining epochs */
    long pretrainEpochs = 0;
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
    /* Use FLOAT data format when parsing input files? */
    bool useFloatDataset = false;
    /* Seed for random generator. */
    int seed = 0;
    /* Path to the file where the network parameters should be exported. */
    char *exportFile = NULL;
    /* Path to the file from which the network parameters should be imported. */
    char *importFile = NULL;
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

void printArray(data_t *arr, int size) {
    
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

/* Runs the given test dataset through given network and prints results. */
void runTest(Network *net, InputDataset *ds) {
    
    ds->reset();
    if (ds->getInputDimension() <= MAX_PRINT_ARRAY_SIZE) {
        while (ds->hasNext()) {
            net->setInput(ds->next());
            net->forward();
            printInout(net);
        }
    } else {

        const SimpleLabeledDataset *LABELED_DATASET_CLASS = new SimpleLabeledDataset(0,0,0);
        LabeledDataset *lds = (LabeledDataset *) ds;
        ErrorComputer *ec = new MseErrorComputer();
        int i = 0;

        while (lds->hasNext()) {
            data_t *pattern = lds->next();
            i++;
            data_t *label = pattern + lds->getInputDimension();
            net->setInput(pattern);
            net->forward();
            if (typeid(*ds)==typeid(*LABELED_DATASET_CLASS)) {
                data_t error = ec->compute(net, label);
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
    
    ds->addPattern((const data_t[2]){0, 0}, (const data_t[1]){0});
    ds->addPattern((const data_t[2]){0, 1}, (const data_t[1]){1});
    ds->addPattern((const data_t[2]){1, 0}, (const data_t[1]){1});
    ds->addPattern((const data_t[2]){1, 1}, (const data_t[1]){0});
    
    return (LabeledDataset*) ds;
}

void printHelp() {
    cout << "Usage: deepframe [OPTIONS]" << endl << endl;
    cout << "Option      GNU long option       Meaning" << endl;
    cout << "--------------------------------------------------------------------------------" << endl;
    cout << "-h          --help                This help." << endl;
    cout << endl;
    cout << "-b          --no-bias             Disables bias in neural network. Bias is enabled by default." << endl;
    cout << endl;
    cout << "-l <value>  --rate <value>        Learning rate influencing the speed and quality of learning. This is a global setting used in case of MLP configured via options. The external layer configuration file overrides this setting and allows to assign a different learning rate for each layer. Default value is 0.3." << endl;
    cout << endl;
    cout << "-a <value>  --init <value>        In case of uniform distribution, minimum and maximum value network weights are initialized to. In case of Gaussian distribution, the standard deviation. Default is uniform distribution with interval (-1,1)." << endl;
    cout << endl;
    cout << "-e <value>  --mse <value>         Target Mean Square Error to determine when to finish the learning. Default is 0.01." << endl;
    cout << endl;
    cout << "-k <value>  --improve-err <value> Number of epochs during which improvement of error is required to keep learning. Default is zero (=disabled)." << endl;
    cout << endl;
    cout << "-m <value>  --max-epochs <value>  Sets a maximum limit for number of epochs. Learning is stopped even if MSE has not been met. Default is 100,000" << endl;
    cout << endl;
    cout << "-n <value>  --pretrain <value>    Configures the number of pretraining epochs for Deep Belief Network. Default is zero (no pretraining)." << endl;
    cout << endl;
    cout << "-f <value>  --func <value>        Specifies the activation function to be used. Use 's' for sigmoid, 'h' for hyperbolic tangent. Sigmoid is the default." << endl;
    cout << endl;
    cout << "-c <value>  --lconf <value>       Specifies layer configuration for the MLP network as a comma separated list of integers. Alternatively, it can contain a path to configuration in an external file. Default value is \"2,2,1\"." << endl;
    cout << endl;
    cout << "-s <value>  --labels <value>      File path with labeled data to be used for learning. For IDX format separate the data and labels filepath with a colon (\":\")." << endl;
    cout << endl;
    cout << "-t <value>  --test <value>        File path with test data to be used for evaluating networks performance. For IDX data with labels for testing dataset separate the data and labels filepath with a colon (\":\")." << endl;
    cout << endl;
    cout << "-v <value>  --validation <value>  Size of the validation set. Patterns are taken from the training set. Default is zero." << endl;
    cout << endl;
    cout << "-q <value>  --k-fold <value>      Number of folds to use in k-fold cross validation. Default is one (=disabled)." << endl;
    cout << endl;
    cout << "-o          --best-fold           Uses the best network trained with k-fold validation. By default epoch limit is averaged and network is trained on all data." << endl;
    cout << endl;
    cout << "-i          --idx                 Use IDX data format when parsing files with datasets. Human readable CSV-like format is the default." << endl;
    cout << endl;
    cout << "-g          --float               Use FLOAT data format when parsing files with datasets. Human readable CSV-like format is the default." << endl;
    cout << endl;
    cout << "-r <value>  --random-seed <value> Specifies value to be used for seeding random generator." << endl;
    cout << endl;
    cout << "-w <value>  --export <value>      Exports the learnt network parameters into the given file." << endl;
    cout << endl;
    cout << "-x <value>  --import <value>      Imports the learnt network parameters from the given file." << endl;
    cout << endl;
    cout << "-j          --shuffle             Shuffles training and validation dataset do the patterns are in random order." << endl;
    cout << endl;
    cout << "-u <value>  --use-cache <value>   Enables use of precomputed lookup table for activation function. Value specifies the size of the table." << endl;
    cout << endl;
    cout << "-p          --use-gpu             Enables parallel implementation of the network using CUDA GPU API." << endl;
    cout << endl;
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
            case 'n' :
                conf->pretrainEpochs = atol(optarg);
                break;
            case 'c' :
                conf->layerConf = new char[strlen(optarg)+1];
                strcpy(conf->layerConf, optarg);
                break;
            case 's' :
                if (strlen(optarg) > 0 && strcmp("xor", optarg) != 0) {
                    conf->labeledData = optarg;
                }
                break;
            case 't' :
                if (strlen(optarg) > 0 && strcmp("xor", optarg) != 0) {
                    conf->testData = optarg;
                }
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
                if (conf->useFloatDataset) {
                    LOG()->warn("Float datasets work with unknown dataset size and hence do not support k-fold validation. Ignoring k-fold validation...");
                } else {
                    conf->useBestFold = true;
                }
                break;
            case 'i' :
                conf->useIdx = true;
                conf->useFloatDataset = false;
                break;
            case 'g' :
                conf->useFloatDataset = true;
                conf->useIdx = false;
                if (conf->useBestFold) {
                    conf->useBestFold = false;
                    LOG()->warn("Float datasets work with unknown dataset size and hence do not support k-fold validation. Ignoring k-fold validation...");
                }
                break;
            case 'r' :
                conf->seed = atoi(optarg);
                break;
            case 'j' :
                conf->shuffle = true;
                break;
            case 'w' :
                if (strlen(optarg) > 0) {
                    conf->exportFile = optarg;
                }
                break;
            case 'x' :
                if (strlen(optarg) > 0) {
                    conf->importFile = optarg;
                }
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
NetworkConfiguration *createNetworkConfiguration(config* conf) {
    
    LOG()->info("Seeding random generator with %d.", conf->seed);
    srand(conf->seed);
    
    NetworkConfiguration *netConf = new NetworkConfiguration();
    
    // Setup function lookup table if it is to be used.
    if (conf->useGpu && conf->useFunctionCache) {
        LOG()->warn("Precomputed activation function table is only supported on CPU, disabling.");
    } else if (conf->useFunctionCache) {
        switch (conf->actFunction) {
            case 's' :
                FunctionCache::init(sigmoidFunction, conf->functionSamples);
                netConf->dActivationFnc = dSigmoidFunction;
                break;
            case 'h' :
                FunctionCache::init(hyperbolicTangentFunction, conf->functionSamples);
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
    netConf->setBias(conf->bias);
    netConf->parseInitInterval(conf->initInterval);
    netConf->setLearningRate(conf->lr);
    netConf->parseLayerConf(conf->layerConf);
    
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

/* Entry point of the application. */
int main(int argc, char *argv[]) {
    
    // prepare network configuration
    config* conf = processOptions(argc, argv);
    
    LOG()->info("Compiled with %dbit precision for data types.", 8*sizeof(data_t));

    // Configure seed for random generator.
    if (conf->seed == 0) {
        conf->seed = getSeed();
    }
    
    NetworkConfiguration *netConf = createNetworkConfiguration(conf);
    
    // construct the network
    Network *net;
    BackpropagationLearner *bp;
    NetworkPretrainer *pretrainer;
    
    // probe GPU and fetch specs
    GpuConfiguration *gpuConf;
    bool useGpu = conf->useGpu;
    if (useGpu) {
        gpuConf = createGpuConfiguration(conf);
        if (gpuConf == NULL) {
            LOG()->warn("Falling back to CPU as GPU probe was unsuccessful.");
            useGpu = false;
        } else {
            netConf->setUseGpu(true);
        }
    }
    
    // setup correct implementations for network and BP learner
    if (useGpu) {
        LOG()->info("Using GPU for computing the network runs.");
        GpuNetwork *gpuNet = new GpuNetwork(netConf, gpuConf);
        pretrainer = new NetworkPretrainer(gpuNet);
        bp = new BackpropagationLearner(gpuNet);
        net = gpuNet;
    } else {
        LOG()->info("Using CPU for computing the network runs.");
        CpuNetwork *cpuNet = new CpuNetwork(netConf);
        pretrainer = new NetworkPretrainer(cpuNet);
        bp = new BackpropagationLearner(cpuNet);
        net = cpuNet;
    }
    
    // Configure network layers
    net->setup();
    
    // Prepare test dataset.
    InputDataset *tds;
    if (conf->testData == NULL) {
        tds = createXorDataset();
    } else if (conf->useIdx) {
        LabeledMnistParser *p = new LabeledMnistParser();
        tds = p->parse(conf->testData);
//        printImageLabels((LabeledDataset *)tds);
//        return 0;
        delete p;
    } else if (conf->useFloatDataset) {
        tds = new FloatDataset(conf->testData);
    } else {
        LabeledDatasetParser *p = new LabeledDatasetParser(conf->testData, netConf);
        tds = p->parse();
        delete p;
    }

    // Run before learning (just to see what we get without learning)
//    runTest(net, tds);
    
    // Prepare training dataset.
    // If none was provided in options use XOR dataset by default.
    LabeledDataset *lds;
    if (conf->labeledData == NULL) {
        lds = createXorDataset();
    } else if (conf->useIdx) {
        LabeledMnistParser *p = new LabeledMnistParser();
        lds = p->parse(conf->labeledData);
        delete p;
    } else if (conf->useFloatDataset) {
        lds = new FloatDataset(conf->labeledData);
    } else {
        LabeledDatasetParser *p = new LabeledDatasetParser(conf->labeledData, netConf);
        lds = p->parse();
        delete p;
    }
    
    // Shuffle labeled data
    if (conf->shuffle) {
        LOG()->info("Randomly shuffling the training dataset.");
        ((InMemoryLabeledDataset *)lds)->shuffle();
        LOG()->info("Random shuffling was successful.");
    }
    
//    printImageLabels((LabeledDataset *)lds);
//    return 0;
    
    // configure learners
    pretrainer->setPretrainEpochs(conf->pretrainEpochs);
    bp->setTargetMse(conf->mse);
    bp->setErrorComputer(new MseErrorComputer());
    bp->setEpochLimit(conf->maxEpochs);
    bp->setImproveEpochs(conf->improveEpochs);
    
    // Prepare validation dataset
    FoldDatasetFactory *df;
    LabeledDataset *vds;
    if (conf->kFold > 1) {
        
        LOG()->info("Training with %d-fold cross validation.", conf->kFold);
        df = new FoldDatasetFactory(((InMemoryLabeledDataset *)lds), conf->kFold);
        data_t bestError = 2.0; // start with impossibly bad error
        data_t bestValIdx = 0;
        Network *bestNet = NULL;
        long totalEpochs = 0;
        
        // train each fold
        for (int i = 0; i<conf->kFold; i++) {
            LabeledDataset *t = (LabeledDataset *) df->getTrainingDataset(i);
            LabeledDataset *v = (LabeledDataset *) df->getValidationDataset(i);
            
            // reinitialize network and its params (weights, bias)
            net->reinit();
            
            // train the network
            pretrainer->pretrain(t);
            TrainingResult *res = bp->train(t, v, i);
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
            pretrainer->pretrain(lds);
            bp->setEpochLimit(totalEpochs / conf->kFold);
            bp->train(lds, vds, 0);
        }

    } else if (conf->validationSize > 0) {
        
        LOG()->info("Picking the last %d samples from the training dataset to be used for validation.");
        vds = ((InMemoryLabeledDataset *)lds)->takeAway(conf->validationSize);
        pretrainer->pretrain(lds);
        bp->train(lds, vds, 0);
        
    } else {
        
        vds = (LabeledDataset *) new SimpleLabeledDataset(0, 0, 0);
        pretrainer->pretrain(lds);
        bp->train(lds, vds, 0);
    }

    if (conf->exportFile != NULL) {
        NetworkSerializer *ns = new NetworkSerializer();
        ns->save(net, conf->exportFile);
        delete ns;
    }

    // Run (hopefully) learnt network.
    runTest(net, tds);
    
    // Release dynamically allocated memory
    if (conf->kFold > 1) {
        delete df;
    }
    
    delete pretrainer;
    delete bp;
    delete lds;
    delete tds;
    delete netConf;
    delete conf;
    delete net;
    return 0;
}

