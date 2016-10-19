/* 
 * File:   GpuNetwork.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "GpuNetwork.h"

#include <cstring>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "GpuConfiguration.h"

//#include "../util/cudaDebugHelpers.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

using namespace std;

GpuNetwork::GpuNetwork(NetworkConfiguration *netConf, GpuConfiguration *gpuConf) : Network(netConf) {
    cublasCreate(&this->cublasHandle);
    this->gpuConf = gpuConf;
}

GpuNetwork::GpuNetwork(const GpuNetwork& orig) : Network(orig.conf) {
    
    // initialize network and allocate memory
    cublasCreate(&this->cublasHandle);
    this->gpuConf = orig.gpuConf;
    
    this->allocateMemory();
    
    // copy data
    int wMemSize = sizeof(data_t) * this->weightsCount;
    int iMemSize = sizeof(data_t) * this->inputsCount;
    checkCudaErrors(cudaMemcpy(this->inputs, orig.inputs, iMemSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->outputDiffs, orig.outputDiffs, iMemSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->weights, orig.weights, wMemSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(this->weightDiffs, orig.weightDiffs, wMemSize, cudaMemcpyDeviceToDevice));
}

GpuNetwork::~GpuNetwork() {
    cublasDestroy(this->cublasHandle);
    cudaFree(this->inputs);
    cudaFree(this->outputDiffs);
    cudaFree(this->weights);
    cudaFree(this->weightDiffs);
    delete[] this->input;
    delete[] this->output;
}

GpuNetwork* GpuNetwork::clone() {
    return new GpuNetwork(*this);
}

void GpuNetwork::merge(Network** nets, int size) {
    
    LOG()->warn("Method for merging CPU networks is not maintained and likely not working correctly.");
    
    int noWeights = this->weightsCount;
    for (int i = 0; i<size; i++) {
        
        // add weights
        k_sumVectors(this->weights, nets[i]->getWeights(), noWeights);
    }
    
    // divide to get the average
    k_divideVector(this->weights, size+1, noWeights);
}

void GpuNetwork::reinit() {
    if (conf->getImportFile() != NULL) {
        LOG()->info("Importing network parameters from file '%s'.", conf->getImportFile());
        ifstream fp(conf->getImportFile(), ios::in|ios::binary);
        if (fp.is_open()) {
            // parse dimension sizes
            data_t *w = new data_t[weightsCount];
            int memSize = weightsCount * sizeof(data_t);
            fp.read((char *) w, memSize);
            fp.close();
            checkCudaErrors(cudaMemcpy(weights, w, memSize, cudaMemcpyHostToDevice));
            delete[] w;
        } else {
            LOG()->error("Cannot open file '%s' for reading network parameters.", conf->getImportFile());
        }
    } else if (conf->getInitMax() < conf->getInitMin()) {
        // cuRAND needs to generate random array of even length
        int size  = weightsCount;
        if (weightsCount % 2 == 1) {
            size++;
        }
        LOG()->info("Randomly initializing weights from Gaussian distribution with standard deviation of %f.", conf->getInitMin());
        CURAND_CHECK(k_generateNormal(*gpuConf->getRandGen(), weights, size, 0., conf->getInitMin()));
    } else {
        LOG()->info("Randomly initializing weights within the interval (%f,%f).", conf->getInitMin(), conf->getInitMax());
        CURAND_CHECK(k_generateUniform(*gpuConf->getRandGen(), weights, weightsCount));
        k_spreadInterval(conf->getInitMin(), conf->getInitMax(), weights, weightsCount);
    }
//    dumpDeviceArray("weights after random init", this->weights, this->weightsCount);
}

void GpuNetwork::allocateMemory() {
    this->input = new data_t[layers[0]->getOutputsCount()];
    this->output = new data_t[layers[noLayers-1]->getOutputsCount()];
    
    checkCudaErrors(cudaMalloc(&this->inputs, sizeof(data_t) * this->inputsCount));
    checkCudaErrors(cudaMalloc(&this->outputDiffs, sizeof(data_t) * this->inputsCount));
    
    // allocate an extra weight, because some CUDA APIs
    // require even array lengths
    checkCudaErrors(cudaMalloc(&this->weights, sizeof(data_t) * (this->weightsCount + 1)));
    checkCudaErrors(cudaMalloc(&this->weightDiffs, sizeof(data_t) * (this->weightsCount + 1)));
    
    this->memExpectedOutput = this->getOutputNeurons() * sizeof(data_t);
    checkCudaErrors(cudaMalloc(&this->expectedOutput, this->memExpectedOutput));
    
    for (int i = 0; i<noLayers; i++) {
        this->layers[i]->cublasHandle = this->cublasHandle;
        this->layers[i]->curandGen = this->gpuConf->getRandGen();
    }
}

void GpuNetwork::setInput(data_t* input) {
//    compare('b', dInputs, inputs, getInputOffset(noLayers));
    processInput(input);
    int memSize = sizeof(data_t) * this->getInputNeurons();
    std::memcpy(this->input, input, memSize);
    checkCudaErrors(cudaMemcpy(this->inputs, input, memSize, cudaMemcpyHostToDevice));
//    compare('a', dInputs, inputs, getInputOffset(noLayers));
    
    // input is always in sync
    inputSynced = true;
}

data_t *GpuNetwork::getInput() {
    return this->input;
}

data_t *GpuNetwork::getOutput() {
    
    if (!outputSynced) {
        // copy network output to host
        data_t *dOutput = this->layers[noLayers-1]->getOutputs();
        int oMemSize = this->getOutputNeurons() * sizeof(data_t);
        checkCudaErrors(cudaMemcpy(this->output, dOutput, oMemSize, cudaMemcpyDeviceToHost));
        outputSynced = true;
    }
    
    return this->output;
}

void GpuNetwork::setExpectedOutput(data_t* output) {
    checkCudaErrors(cudaMemcpy(expectedOutput, output, this->memExpectedOutput, cudaMemcpyHostToDevice));
}

void GpuNetwork::forward() {
    for (int i = 1; i < this->noLayers; i++) {
        LOG()->debug("Computing forward run on GPU for layer %d.", i);
        this->layers[i]->forwardGpu();
    }
    
    // mark output as out of sync
    outputSynced = false;
}

void GpuNetwork::backward() {
    
    LOG()->debug("Computing output gradients for last layer on GPU.");
    
    // clear weightDiffs from previous runs
    cudaMemset(weightDiffs, 0, weightsCount * sizeof(data_t));
    
    this->layers[noLayers-1]->backwardLastGpu(expectedOutput);
    
    for (int i = noLayers-1; i > 0; i--) {
        LOG()->debug("Computing backward run on GPU for layer %d.", i);
        this->layers[i]->backwardGpu();
    }
    
    // update all weights
    k_sumVectors(weights, weightDiffs, weightsCount);
}

void GpuNetwork::save(char *filePath) {

    // open IDX file with the dataset
    ofstream fp(filePath, ios::out|ios::binary);

    if (fp.is_open()) {

        // copy weights to host
        int memSize = weightsCount * sizeof(data_t);
        data_t *w = new data_t[weightsCount];
        checkCudaErrors(cudaMemcpy(w, weights, memSize, cudaMemcpyDeviceToHost));

        // write weights
        for (int i = 0; i<weightsCount; i++) {
            fp.write((char *) (w+i), sizeof(data_t));
        }

        fp.close();
        delete[] w;

        LOG()->info("Serialized network parameters in file '%s'.", filePath);
    } else {
        LOG()->error("Cannot open file '%s' for writing.", filePath);
    }
}
