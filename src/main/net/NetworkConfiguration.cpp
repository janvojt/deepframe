/* 
 * File:   NetworkConfiguration.cpp
 * Author: Jan Vojt
 * 
 * Created on June 5, 2014, 8:16 PM
 */

#include "NetworkConfiguration.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cstring>

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

NetworkConfiguration::NetworkConfiguration() {
    bias = true;
}

NetworkConfiguration::NetworkConfiguration(const NetworkConfiguration& orig) {
}

NetworkConfiguration::~NetworkConfiguration() {
    
    if (layersConf) {
        for (int i = 0; i<layers; i++) {
            delete[] layersConf[i];
        }
        delete[] layersConf;
    }
}

int NetworkConfiguration::getLayers() {
    return layers;
}

void NetworkConfiguration::setLayers(int layers) {
    if (layers < 1) {
        throw std::invalid_argument("Number of layers must be a natural number.");
    } else {
        this->layers = layers;
    }
}

string NetworkConfiguration::getLayerType(int layerIndex) {
    return layersConf[layerIndex][0];
}

string NetworkConfiguration::getLayersConf(int layerIndex) {
    return layersConf[layerIndex][1];
}

void NetworkConfiguration::setBias(bool enabled) {
    bias = enabled;
}

bool NetworkConfiguration::getBias() {
    return bias;
}

void NetworkConfiguration::initConf() {
    layersConf = new string*[layers];
    for (int i = 0; i<layers; i++) {
        layersConf[i] = new string[2];
    }
}

void NetworkConfiguration::parseInitInterval(const char* intervalConf) {

#ifdef USE_64BIT_PRECISION
    parseInitInterval(intervalConf, "%lf");
#else
    parseInitInterval(intervalConf, "%f");
#endif    
}

void NetworkConfiguration::parseInitInterval(const char* intervalConf, const char *format) {
    
    data_t v = 0;
    char *haystack = new char[strlen(intervalConf)+1];
    strcpy(haystack, intervalConf);
    char *token = strtok(haystack, ",");
    sscanf(token, format, &v);
    this->initMin = v;
    token = strtok(NULL, ",");

    if (token == NULL) {
        // only a single decimal is given
        // -> we will use Gaussian distribution, store the standard deviation in
        //    initMin and disable initMax by setting to -1.
        this->initMax = -1;
    } else {
        // interval is given
        // -> we will initialize weights from uniform distribution with given
        //    interval
        sscanf(token, format, &v);
        this->initMax = v;
    }
    
    delete[] haystack;
}

void NetworkConfiguration::parseLayerConf(char* layerConf) {
    
    this->confSource = layerConf;
    if (strstr(layerConf, ",") == NULL) {
        parseFromFile(layerConf);
    } else {
        parseFromString(layerConf);
    }
}

void NetworkConfiguration::parseFromString(char *confString) {

    // Configure layers.
    // Count and set number of layers.
    char *lconf = confString;
    layers = 0;
    for (int i=0; lconf[i]; i++) {
        if (lconf[i] == ',') layers++;
    }
    layers++;
    initConf();
    
    // set number of neurons for each layer
    int i = 0;
    char *haystack = new char[strlen(confString)+1];
    strcpy(haystack, confString);
    char *token = strtok(haystack, ",");
    const string sep = ":";
    string confStr = this->bias ? "true" : "false";
    while (token != NULL) {
        layersConf[i][0] = "FullyConnected";
        layersConf[i][1] = token + sep + std::to_string(learningRate) + sep + confStr;
        i++;
        token = strtok(NULL, ",");
    }
    delete[] haystack;
}

void NetworkConfiguration::parseFromFile(char* confFile) {
    
    ifstream infile(confFile);
    string line;
    
    // find out number of layers first
    layers = 0;
    while (getline(infile, line)) {
        if (line.length() == 0 || line[0] == '#') {
            // line is empty or is a comment
            continue;
        }
        layers++;
    }
    initConf();
    
    // parse layer configuration on second pass
    infile.clear();
    infile.seekg(0, ios::beg);
    int i = 0;
    while (getline(infile, line)) {
        if (line.length() == 0 || line[0] == '#') {
            // line is empty or is a comment
            continue;
        }
        int delPos = line.find(':');
        layersConf[i][0] = line.substr(0, delPos);
        layersConf[i][1] = line.substr(delPos + 1);
        i++;
    }
}

void NetworkConfiguration::setInitMin(data_t min) {
    initMin = min;
}

data_t NetworkConfiguration::getInitMin() {
    return initMin;
}

void NetworkConfiguration::setInitMax(data_t max) {
    initMax = max;
}

data_t NetworkConfiguration::getInitMax() {
    return initMax;
}

char* NetworkConfiguration::getConfSource() {
    return confSource;
}

data_t NetworkConfiguration::getLearningRate() {
    return this->learningRate;
}

void NetworkConfiguration::setLearningRate(data_t learningRate) {
    this->learningRate = learningRate;
}

char *NetworkConfiguration::getImportFile() {
    return importFile;
}

void NetworkConfiguration::setImportFile(char *importFile) {
     this->importFile = importFile;
}

bool NetworkConfiguration::getUseGpu() {
    return useGpu;
}

void NetworkConfiguration::setUseGpu(bool useGpu) {
    this->useGpu = useGpu;
}
